"""Paper-faithful, model-independent implementation of DISC.

The implementation follows Algorithms 3 and 4 and Equations (20)--(26) of
Kordi Boroujeny et al., "Multi-Bit Distortion-Free Watermarking for Large
Language Models" (arXiv:2402.16578v1).

All logarithms are natural logarithms, as in the paper.  Positions in this
module are zero based; ``n_star`` is the number of random-initialization bits.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaincc

MessageMapping = Literal["direct", "gray"]


def gray_encode(value: int) -> int:
    """Map a non-negative integer to its binary-reflected Gray code."""
    if value < 0:
        raise ValueError("value must be non-negative")
    return value ^ (value >> 1)


def gray_decode(value: int) -> int:
    """Invert a binary-reflected Gray-code integer."""
    if value < 0:
        raise ValueError("value must be non-negative")
    decoded = 0
    while value:
        decoded ^= value
        value >>= 1
    return decoded


def _map_payload(payload: int, mapping: MessageMapping) -> int:
    if mapping == "direct":
        return payload
    if mapping == "gray":
        return gray_encode(payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _unmap_payload(mapped_payload: int, mapping: MessageMapping) -> int:
    if mapping == "direct":
        return mapped_payload
    if mapping == "gray":
        return gray_decode(mapped_payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _validate_bits(bits: Sequence[int]) -> None:
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


class HmacPrf:
    """HMAC-SHA256 PRF with an explicit, portable input encoding."""

    domain = b"DISC-v1\x00"

    def __init__(self, key: bytes | str | int):
        if isinstance(key, str):
            key = key.encode("utf-8")
        elif isinstance(key, int):
            if key < 0:
                raise ValueError("integer keys must be non-negative")
            key = key.to_bytes(max(1, (key.bit_length() + 7) // 8), "big")
        if not isinstance(key, bytes) or not key:
            raise ValueError("key must be a non-empty bytes, str, or integer value")
        self._key = key

    @staticmethod
    def _pack_bits(bits: Sequence[int]) -> bytes:
        _validate_bits(bits)
        packed = bytearray((len(bits) + 7) // 8)
        for i, bit in enumerate(bits):
            packed[i // 8] |= bit << (7 - i % 8)
        return struct.pack(">I", len(bits)) + bytes(packed)

    def uniform(self, initial_bits: Sequence[int], context_bits: Sequence[int]) -> float:
        message = self.domain + self._pack_bits(initial_bits) + self._pack_bits(context_bits)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()
        # Midpoint mapping avoids returning exactly 0 or 1.
        integer = int.from_bytes(digest[:8], "big")
        return (integer + 0.5) / 2**64


def conditional_bit_probability(
    probabilities: Sequence[float] | np.ndarray,
    prefix: int,
    bit_index: int,
    bit_length: int | None = None,
) -> float:
    """Return P(next binary token = 1 | current token prefix).

    This implements the binarized LM in Equations (31)--(32). Token IDs are
    represented with ``ceil(log2(vocab_size))`` bits; nonexistent leaves have
    zero probability.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError("probabilities must be a one-dimensional vocabulary distribution")
    if np.any(probs < 0) or not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("probabilities must be finite, non-negative, and have positive mass")
    probs = probs / probs.sum()
    width = bit_length or math.ceil(math.log2(probs.size))
    if not 0 <= bit_index < width:
        raise ValueError("bit_index is outside the token representation")
    if not 0 <= prefix < 2**bit_index:
        raise ValueError("prefix is incompatible with bit_index")

    remaining = width - bit_index - 1
    zero_start = (prefix << 1) << remaining
    one_start = ((prefix << 1) | 1) << remaining
    span = 1 << remaining
    zero_mass = probs[zero_start : min(zero_start + span, probs.size)].sum()
    one_mass = probs[one_start : min(one_start + span, probs.size)].sum()
    total = zero_mass + one_mass
    return float(one_mass / total) if total > 0 else 0.0


def token_ids_to_bits(token_ids: Sequence[int], bit_length: int) -> list[int]:
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    result: list[int] = []
    for token_id in token_ids:
        if not 0 <= token_id < 2**bit_length:
            raise ValueError(f"token id {token_id} does not fit in {bit_length} bits")
        result.extend(int(bit) for bit in f"{token_id:0{bit_length}b}")
    return result


def bits_to_token_ids(bits: Sequence[int], bit_length: int) -> list[int]:
    _validate_bits(bits)
    if bit_length <= 0 or len(bits) % bit_length:
        raise ValueError("bit sequence length must be divisible by bit_length")
    return [
        int("".join(map(str, bits[i : i + bit_length])), 2) for i in range(0, len(bits), bit_length)
    ]


def in_shifted_interval(y: float, probability_one: float, delta_m: float) -> bool:
    """Membership in A_2(M), Equations (21)--(22), on the unit circle."""

    if not 0.0 <= y < 1.0 or not 0.0 <= probability_one <= 1.0:
        raise ValueError("y and probability_one must lie in [0, 1]")
    start = delta_m % 1.0
    end = start + probability_one
    return start <= y < end if end <= 1.0 else y >= start or y < end - 1.0


def disc_score(bit: int, y: float, delta_m: float) -> float:
    """DISC score from Equation (24)."""

    if bit not in (0, 1) or not 0.0 <= y < 1.0:
        raise ValueError("bit must be binary and y must lie in [0, 1)")
    distance = ((y - delta_m) % 1.0) if bit else ((delta_m - y) % 1.0)
    # PRF midpoint output makes zero probability negligible; guard exact test inputs.
    return -math.log(max(distance, np.finfo(np.float64).tiny))


class DiscEncoder:
    """Stateful DISC binary-token encoder with random initialization."""

    def __init__(
        self,
        key: bytes | str | int,
        payload: int,
        payload_bits: int,
        *,
        entropy_threshold: float = 5.0,
        context_width: int = 16,
        seed: int | None = None,
        message_mapping: MessageMapping = "direct",
    ):
        if payload_bits < 1 or not 0 <= payload < 2**payload_bits:
            raise ValueError("payload must be in [0, 2**payload_bits)")
        if entropy_threshold < 0 or context_width < 1:
            raise ValueError("entropy_threshold must be non-negative and context_width positive")
        self.prf = HmacPrf(key)
        self.payload = payload
        self.payload_bits = payload_bits
        self.message_mapping = message_mapping
        self.mapped_payload = _map_payload(payload, message_mapping)
        self.delta_m = self.mapped_payload / 2**payload_bits
        self.entropy_threshold = entropy_threshold
        self.context_width = context_width
        self.rng = random.Random(seed)
        self.bits: list[int] = []
        self.empirical_entropy = 0.0
        self.n_star: int | None = None

    @property
    def random_initialization(self) -> list[int]:
        end = len(self.bits) if self.n_star is None else self.n_star
        return self.bits[:end]

    def encode_bit(self, probability_one: float) -> int:
        if not 0.0 <= probability_one <= 1.0:
            raise ValueError("probability_one must lie in [0, 1]")
        if self.n_star is None and (
            self.empirical_entropy < self.entropy_threshold or len(self.bits) < self.context_width
        ):
            bit = int(self.rng.random() < probability_one)
            chosen_probability = probability_one if bit else 1.0 - probability_one
            self.empirical_entropy += -math.log(max(chosen_probability, np.finfo(float).tiny))
            self.bits.append(bit)
            if (
                self.empirical_entropy >= self.entropy_threshold
                and len(self.bits) >= self.context_width
            ):
                self.n_star = len(self.bits)
            return bit

        if self.n_star is None:
            self.n_star = len(self.bits)
        context = self.bits[-self.context_width :]
        y = self.prf.uniform(self.bits[: self.n_star], context)
        bit = int(in_shifted_interval(y, probability_one, self.delta_m))
        self.bits.append(bit)
        return bit

    def encode_token(self, probabilities: Sequence[float] | np.ndarray) -> int:
        probs = np.asarray(probabilities, dtype=np.float64)
        width = math.ceil(math.log2(probs.size))
        prefix = 0
        for bit_index in range(width):
            p_one = conditional_bit_probability(probs, prefix, bit_index, width)
            prefix = (prefix << 1) | self.encode_bit(p_one)
        if prefix >= probs.size:
            raise RuntimeError("binarized sampling reached a zero-probability token")
        return prefix


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    payload: int | None
    n_star: int | None
    local_p_value: float
    global_p_value: float
    score: float
    scored_bits: int


class DiscDetector:
    """DISC detector implementing the search and correction in Algorithm 4."""

    def __init__(
        self,
        key: bytes | str | int,
        payload_bits: int,
        *,
        context_width: int = 16,
        fpr: float = 0.01,
        deduplicate_ngrams: bool = True,
        message_mapping: MessageMapping = "direct",
    ):
        if payload_bits < 1 or context_width < 1:
            raise ValueError("payload_bits and context_width must be positive")
        if not 0 < fpr < 1:
            raise ValueError("fpr must lie strictly between 0 and 1")
        self.prf = HmacPrf(key)
        self.payload_bits = payload_bits
        self.message_count = 2**payload_bits
        self.context_width = context_width
        self.fpr = fpr
        self.deduplicate_ngrams = deduplicate_ngrams
        _map_payload(0, message_mapping)
        self.message_mapping = message_mapping

    def _score(self, bits: Sequence[int], n_star: int, payload: int) -> tuple[float, int]:
        initial = bits[:n_star]
        delta_m = payload / self.message_count
        score = 0.0
        count = 0
        seen: set[tuple[int, ...]] = set()
        for index in range(n_star, len(bits)):
            context = bits[index - self.context_width : index]
            ngram = tuple(context) + (bits[index],)
            if self.deduplicate_ngrams and ngram in seen:
                continue
            seen.add(ngram)
            y = self.prf.uniform(initial, context)
            score += disc_score(bits[index], y, delta_m)
            count += 1
        return score, count

    def detect(
        self, bits: Sequence[int], *, n_star_candidates: Iterable[int] | None = None
    ) -> DetectionResult:
        bits = list(bits)
        _validate_bits(bits)
        if len(bits) <= self.context_width:
            return DetectionResult(False, None, None, 1.0, 1.0, 0.0, 0)
        candidates = (
            list(n_star_candidates)
            if n_star_candidates is not None
            else range(self.context_width, len(bits))
        )
        best = (1.0, 0.0, 0, None, None)  # local p, score, count, n*, payload
        for n_star in candidates:
            if not self.context_width <= n_star < len(bits):
                continue
            for payload in range(self.message_count):
                score, count = self._score(bits, n_star, payload)
                local_p = float(gammaincc(count, score)) if count else 1.0
                if local_p < best[0]:
                    best = (local_p, score, count, n_star, payload)

        local_p, score, count, n_star, mapped_payload = best
        # Equation (26): union correction across messages and candidate starts.
        per_start = min(1.0, self.message_count * local_p)
        global_p = (
            -math.expm1((len(bits) - self.context_width) * math.log1p(-per_start))
            if per_start < 1.0
            else 1.0
        )
        detected = global_p <= self.fpr
        return DetectionResult(
            detected,
            _unmap_payload(mapped_payload, self.message_mapping) if detected else None,
            n_star,
            local_p,
            global_p,
            score,
            count,
        )
