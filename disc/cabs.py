"""Context-Anchored Balanced Scheduler (CABS) from MirrorMark Algorithm 1.

CABS assigns each eligible token to one of ``H`` message positions. It is used
here as a scheduler for DISC, not as MirrorMark: each assigned position is
watermarked with the DISC shifted-interval rule for that position's symbol.

Two mechanisms (Jiang et al., arXiv:2601.22246v3, Appendix F):

1. Balanced allocation: send the next eligible token to a currently
   least-populated position. Ties are broken by ``PRF_sk(last h tokens)``.
2. Context-anchored frames: a sliding window ``Q`` of ``W`` assigned tokens
   is hashed. A new frame starts when the ``f`` LSBs of that hash are zero
   (and the frame is at least ``min_len`` long), or when the frame reaches
   ``max_len = max_factor * H``. Counts reset at each new frame so a local
   insertion/deletion cannot desynchronize the whole sequence.

Example::

    from disc.cabs import CabsConfig, CabsScheduler
    from disc.core import HmacPrf

    scheduler = CabsScheduler(HmacPrf("secret"), n_positions=4, context_width=5)
    positions = scheduler.assign_sequence([11, 7, 3, 9, 2, 8])
    # positions[t] is int in {0,1,2,3} or None if token t was ineligible
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class _CabsPrf(Protocol):
    """PRF methods CABS needs. Implemented by ``disc.core.HmacPrf``."""

    domain_cabs_frame: bytes
    domain_cabs_pos: bytes

    def integer_hash(self, tokens: Sequence[int], *, domain: bytes) -> int: ...

    def uniform_tokens(self, tokens: Sequence[int], *, domain: bytes) -> float: ...


@dataclass(frozen=True)
class CabsConfig:
    """CABS hyperparameters. Defaults match the MirrorMark ablation.

    Attributes:
        window_size: Frame-hash window ``W``, ``int``. Default ``4``.
            ``Q`` holds the last ``W`` *assigned* token IDs.
        frame_bits: ``f``, ``int``. A hash-triggered cut occurs when
            ``Hash(Q) mod 2**f == 0``, i.e. probability ``2**(-f)``.
            Default ``3`` (cut chance 1/8 once ``min_len`` is met).
        max_factor: ``float``. ``max_len = max(H, floor(max_factor * H))``.
            Default ``1.5``.
        min_len: Minimum assigned tokens before a hash cut is allowed.
            ``None`` (default) uses ``H``. Example: ``H=4`` → wait 4 tokens.
    """

    window_size: int = 4
    frame_bits: int = 3
    max_factor: float = 1.5
    min_len: int | None = None


class CabsScheduler:
    """Stateful CABS assigner following Algorithm 1.

    Call ``propose(history)`` *before* sampling the next token, then
    ``commit(token_id)`` after it is known. ``assign_sequence`` replays both
    steps over a finished token list (encoder and detector must match).

    Args:
        prf: Object with ``uniform_tokens`` and ``integer_hash`` (``HmacPrf``).
        n_positions: ``H``, number of payload symbols. Example: ``4``.
        context_width: ``h``, token n-gram length for Elig and tie-breaking.
            Example: ``5``.
        config: Optional ``CabsConfig``. ``None`` uses paper defaults.
    """

    def __init__(
        self,
        prf: _CabsPrf,
        n_positions: int,
        context_width: int,
        config: CabsConfig | None = None,
    ):
        if n_positions < 1:
            raise ValueError("n_positions must be positive")
        if context_width < 1:
            raise ValueError("context_width must be positive")
        self.prf = prf
        self.n_positions = n_positions
        self.context_width = context_width
        self.config = config or CabsConfig()
        if self.config.window_size < 1 or self.config.frame_bits < 1 or self.config.max_factor <= 0:
            raise ValueError("CABS window_size, frame_bits must be positive and max_factor > 0")
        self.min_len = self.config.min_len if self.config.min_len is not None else n_positions
        if self.min_len < 1:
            raise ValueError("min_len must be positive")
        self.max_len = max(n_positions, math.floor(self.config.max_factor * n_positions))
        self.reset()

    def reset(self) -> None:
        """Clear frame counts, the hash window, and the eligibility set."""
        self.counts: list[int] = [0] * self.n_positions
        self.queue: list[int] = []  # last W assigned token IDs, oldest first
        self.frame_len = 0  # assigned tokens in the current frame
        self.seen: set[tuple[int, ...]] = set()  # h-grams already used to watermark
        self._pending: tuple[tuple[int, ...], int, int] | None = None

    def propose(self, history: Sequence[int]) -> int | None:
        """Return a position for the next token, or None if it should not be watermarked.

        Args:
            history: Token IDs already generated, ``Sequence[int]``.
                Example: ``[11, 7, 3, 9]``. Must *not* include the token being
                sampled. Uses the last ``h`` IDs as context.

        Returns:
            ``int`` in ``{0, ..., H-1}`` if eligible, else ``None``.
            Warm-up (fewer than ``h`` tokens) and repeated h-grams are
            ineligible, matching ``Elig`` in Algorithm 1.
        """
        if len(history) < self.context_width:
            self._pending = None
            return None
        context = tuple(history[-self.context_width :])
        if context in self.seen:
            self._pending = None
            return None
        # Algorithm 1 hashes Q *before* enqueueing the current token.
        frame_hash = self.prf.integer_hash(self.queue, domain=self.prf.domain_cabs_frame)
        position = self._choose_position(context)
        self._pending = (context, position, frame_hash)
        return position

    def commit(self, token_id: int) -> None:
        """Record a generated token ID. No-op if the last ``propose`` was ineligible.

        Args:
            token_id: Vocabulary ID just sampled, ``int``. Example: ``318``.
        """
        if self._pending is None:
            return
        context, position, frame_hash = self._pending
        self._pending = None
        self.seen.add(context)
        self.counts[position] += 1
        self.frame_len += 1
        self.queue.append(int(token_id))
        if len(self.queue) > self.config.window_size:
            self.queue.pop(0)
        hash_cut = (
            self.frame_len >= self.min_len
            and frame_hash % (1 << self.config.frame_bits) == 0
        )
        if hash_cut or self.frame_len >= self.max_len:
            self.counts = [0] * self.n_positions
            self.queue = []
            self.frame_len = 0

    def assign_sequence(self, token_ids: Sequence[int]) -> list[int | None]:
        """Replay CABS over a finished token sequence.

        Args:
            token_ids: Full ID list, e.g. ``[11, 7, 3, 9, 2]``.

        Returns:
            ``list[int | None]`` of length ``len(token_ids)``. Index ``t`` is
            the position used while generating token ``t``, or ``None`` if
            that token was not watermarked.
        """
        self.reset()
        positions: list[int | None] = []
        for index, token_id in enumerate(token_ids):
            positions.append(self.propose(token_ids[:index]))
            self.commit(int(token_id))
        return positions

    def _choose_position(self, context: Sequence[int]) -> int:
        """Least-populated position; PRF tie-break among argmin slots."""
        least = min(self.counts)
        candidates = [index for index, count in enumerate(self.counts) if count == least]
        if len(candidates) == 1:
            return candidates[0]
        # pos ~ Unif(argmin) seeded by PRF_sk(x_{i-h:i-1}).
        unit = self.prf.uniform_tokens(context, domain=self.prf.domain_cabs_pos)
        return candidates[int(unit * len(candidates)) % len(candidates)]
