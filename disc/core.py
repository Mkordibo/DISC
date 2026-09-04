"""Paper-faithful, model-independent implementation of DISC.

The implementation follows Algorithms 3 and 4 and Equations (20)--(26) of
Kordi Boroujeny et al., "Multi-Bit Distortion-Free Watermarking for Large
Language Models" (arXiv:2402.16578v1).

All logarithms are natural logarithms, as in the paper. Positions in this
module are zero based; ``n_star`` is the number of random-initialization bits.

Data the rest of the package works with
---------------------------------------
- ``bits``: ``list[int]`` of 0/1 values, e.g. ``[1, 0, 1, 1, 0]``.
- ``payload``: ``int`` in ``{0, ..., 2**payload_bits - 1}``, e.g. ``5`` with
  ``payload_bits=3`` (eight possible messages).
- ``delta_m``: ``float`` shift on the unit interval, ``payload / 2**m``.
  Example: payload 5, m=3 → ``5 / 8 = 0.625``.
- ``probability_one`` / ``p_one``: ``float`` in ``[0, 1]``, P(next bit = 1).
- ``y``: ``float`` in ``[0, 1)``, HMAC-PRF output used as a uniform sample.
- ``n_star``: ``int``, length of the random (unwatermarked) prefix.
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

from .cabs import CabsConfig, CabsScheduler

# "direct" embeds payload M as delta = M / 2^m (paper default).
# "gray" first maps M to binary-reflected Gray code, then uses that index.
# Example: payload 6 with m=3 → direct uses 6; gray uses gray_encode(6) = 5.
MessageMapping = Literal["direct", "gray"]

# How the PRF is seeded for each binary token (DISC §3.2 vs §3.3, plus token n-grams).
#   "prefix_bit_ngram"    — y = F(R, last h bits). Default DISC with random init.
#   "bit_ngram"           — y = F(last h bits). DISC without random initialization.
#   "token_ngram"         — y = F(last h token IDs, bit index). Natural with CABS.
#   "prefix_token_ngram"  — y = F(random-init token prefix R, last h token IDs, bit index).
ContextMode = Literal["prefix_bit_ngram", "bit_ngram", "token_ngram", "prefix_token_ngram"]


def gray_encode(value: int) -> int:
    """Map a non-negative integer to its binary-reflected Gray code.

    Args:
        value: Non-negative integer. Example: ``6`` (binary ``110``).

    Returns:
        Gray-coded integer. Example: ``gray_encode(6) == 5`` because
        ``6 ^ (6 >> 1) == 6 ^ 3 == 5`` (binary ``110 ^ 011 = 101``).

        For ``range(8)`` the map is ``[0, 1, 3, 2, 6, 7, 5, 4]``.
    """
    if value < 0:
        raise ValueError("value must be non-negative")
    return value ^ (value >> 1)


def gray_decode(value: int) -> int:
    """Invert a binary-reflected Gray-code integer.

    Args:
        value: Gray-coded non-negative integer. Example: ``5``.

    Returns:
        Original integer. Example: ``gray_decode(5) == 6``.
        Always ``gray_decode(gray_encode(n)) == n`` for ``n >= 0``.
    """
    if value < 0:
        raise ValueError("value must be non-negative")
    decoded = 0
    # XOR successive right-shifts: 5 (101) → 5 ^ 2 ^ 1 = 6.
    while value:
        decoded ^= value
        value >>= 1
    return decoded


def _map_payload(payload: int, mapping: MessageMapping) -> int:
    """Convert a user payload to the interval index actually embedded.

    Args:
        payload: Integer message, e.g. ``5``.
        mapping: ``"direct"`` leaves it unchanged; ``"gray"`` Gray-encodes it.

    Returns:
        Interval index as ``int``. Examples:
        ``_map_payload(6, "direct") == 6``;
        ``_map_payload(6, "gray") == 5``.
    """
    if mapping == "direct":
        return payload
    if mapping == "gray":
        return gray_encode(payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _unmap_payload(mapped_payload: int, mapping: MessageMapping) -> int:
    """Invert ``_map_payload`` after the detector recovers an interval index.

    Args:
        mapped_payload: Index the detector scored, e.g. ``5``.
        mapping: Must match the encoder. ``"direct"`` or ``"gray"``.

    Returns:
        User-facing payload as ``int``. Examples:
        ``_unmap_payload(6, "direct") == 6``;
        ``_unmap_payload(5, "gray") == 6``.
    """
    if mapping == "direct":
        return mapped_payload
    if mapping == "gray":
        return gray_decode(mapped_payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _validate_bits(bits: Sequence[int]) -> None:
    """Raise if ``bits`` is not a sequence of 0/1 integers.

    Args:
        bits: Sequence of integers, e.g. ``[0, 1, 1, 0]``. Values like
        ``2`` or ``-1`` are rejected.
    """
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


def split_payload(payload: int, n_positions: int, symbol_bits: int) -> list[int]:
    """Split an integer payload into ``H`` big-endian ``symbol_bits``-wide symbols.

    Args:
        payload: Integer in ``[0, 2**(H * symbol_bits))``. Example: ``27``.
        n_positions: ``H``. Example: ``2``.
        symbol_bits: Bits per position ``m``. Example: ``4`` → symbols in 0..15.

    Returns:
        ``list[int]`` of length ``H``. Example: payload ``27``, ``H=2``,
        ``m=4`` → ``[1, 11]`` because ``27 = 0b0001_1011``.
    """
    if n_positions < 1 or symbol_bits < 1:
        raise ValueError("n_positions and symbol_bits must be positive")
    mask = (1 << symbol_bits) - 1
    symbols = []
    value = payload
    for _ in range(n_positions):
        symbols.append(value & mask)
        value >>= symbol_bits
    if value:
        raise ValueError("payload does not fit in n_positions * symbol_bits")
    symbols.reverse()
    return symbols


def join_payload(symbols: Sequence[int], symbol_bits: int) -> int:
    """Inverse of ``split_payload``.

    Args:
        symbols: Per-position integers, e.g. ``[1, 11]``.
        symbol_bits: ``m``. Example: ``4``.

    Returns:
        Combined integer. Example: ``join_payload([1, 11], 4) == 27``.
    """
    payload = 0
    for symbol in symbols:
        payload = (payload << symbol_bits) | int(symbol)
    return payload


def combine_p_values(p_values: Sequence[float]) -> float:
    """Fisher combination of independent p-values, keeping a single FPR level.

    If each ``p_h`` is Uniform(0, 1) under H0, then
    ``-sum(log p_h)`` is Erlang(``H``) and the combined p-value is
    ``gammaincc(H, -sum(log p_h))``. Comparing that to the same ``fpr`` as
    single-position DISC controls the overall false-positive rate.

    Args:
        p_values: Per-position p-values, each ``float`` in ``(0, 1]``.
            Example: ``[0.04, 0.10, 0.02]``. Empty → ``1.0``.

    Returns:
        Combined ``float`` p-value in ``[0, 1]``.
    """
    values = [min(1.0, max(float(p), np.finfo(np.float64).tiny)) for p in p_values]
    if not values:
        return 1.0
    score = -sum(math.log(p) for p in values)
    return float(gammaincc(len(values), score))


def prf_uniform(
    prf: HmacPrf,
    context_mode: ContextMode,
    *,
    bits: Sequence[int],
    bit_index: int,
    context_width: int,
    n_star: int = 0,
    tokens: Sequence[int] | None = None,
    token_index: int | None = None,
    bit_in_token: int = 0,
    n_star_tokens: int = 0,
) -> float:
    """Draw the DISC PRF sample ``y`` for one binary token.

    Args:
        prf: Shared ``HmacPrf``.
        context_mode: Which strings seed the PRF. See ``ContextMode``.
        bits: Full bit stream so far (encode) or the whole stream (detect).
        bit_index: Index of the bit being generated/scored. Context is
            ``bits[bit_index - h : bit_index]``.
        context_width: ``h``.
        n_star: Random-init bit length for prefix modes.
        tokens: Token IDs, required for token n-gram modes.
        token_index: Index of the current real token (not yet including it).
        bit_in_token: Bit offset inside that token, ``0`` = MSB.
        n_star_tokens: Random-init token length for ``prefix_token_ngram``.

    Returns:
        ``float`` in ``(0, 1)``.
    """
    if context_mode in ("prefix_bit_ngram", "bit_ngram"):
        start = bit_index - context_width
        if start < 0:
            raise ValueError("bit_index is too small for context_width")
        context = bits[start:bit_index]
        initial = bits[:n_star] if context_mode == "prefix_bit_ngram" else []
        return prf.uniform(initial, context)
    if tokens is None or token_index is None:
        raise ValueError("token n-gram context requires tokens and token_index")
    if token_index < context_width:
        raise ValueError("token_index is too small for context_width")
    history = tokens[token_index - context_width : token_index]
    extra = [(bit_in_token >> shift) & 1 for shift in range(7, -1, -1)]
    if context_mode == "token_ngram":
        return prf.uniform_tokens(history, extra)
    if context_mode == "prefix_token_ngram":
        return prf.uniform_tokens(list(tokens[:n_star_tokens]) + list(history), extra)
    raise ValueError("context_mode must be a ContextMode literal")


class HmacPrf:
    """HMAC-SHA256 PRF with an explicit, portable input encoding.

    Maps a secret key plus two bit-strings (random prefix ``R`` and n-gram
    context ``S``) to a uniform sample ``y`` in ``(0, 1)``.

    Args / attributes:
        key: Secret. Accepted types:
            - ``str``, e.g. ``"secret"`` (UTF-8 encoded);
            - ``bytes``, e.g. ``b"secret"``;
            - non-negative ``int``, e.g. ``42`` (big-endian bytes).

    Example::

        prf = HmacPrf("secret")
        y = prf.uniform(initial_bits=[1, 0, 1], context_bits=[0, 1, 1, 0])
        # y is float, e.g. 0.137..., always in (0, 1), never exactly 0 or 1
    """

    # Domain separators so other HMAC uses of the same key cannot collide.
    domain = b"DISC-v1\x00"
    domain_tokens = b"DISC-v1-tok\x00"
    domain_cabs_pos = b"DISC-v1-cabs-pos\x00"
    domain_cabs_frame = b"DISC-v1-cabs-frame\x00"

    def __init__(self, key: bytes | str | int):
        # HMAC-SHA256 only accepts a bytes key. Callers may pass str, int, or
        # bytes, so normalize everything to non-empty ``bytes`` first.
        #
        #   HmacPrf("secret")  →  b"secret"
        #   HmacPrf(b"secret") →  b"secret"   (already bytes; skip conversion)
        #   HmacPrf(42)        →  b"*"        (42 as one big-endian byte)
        if isinstance(key, str):
            # Text → UTF-8 bytes. "secret" is 6 chars → b"secret" (6 bytes).
            key = key.encode("utf-8")
        elif isinstance(key, int):
            if key < 0:
                raise ValueError("integer keys must be non-negative")
            # How many bytes are needed to hold this integer?
            # bit_length = number of bits in the binary representation.
            #   42  → 0b101010 → 6 bits  → ceil(6/8) = 1 byte  → b"*"  (0x2A)
            #   256 → 0b100000000 → 9 bits → ceil(9/8) = 2 bytes → b"\x01\x00"
            # ``max(1, ...)`` keeps 0 from becoming a 0-length (empty) key.
            n_bytes = max(1, (key.bit_length() + 7) // 8)
            key = key.to_bytes(n_bytes, "big")
        # If the caller passed something else (e.g. None, a list) it is still
        # not bytes here. Also reject b"" — HMAC needs a non-empty secret.
        if not isinstance(key, bytes) or not key:
            raise ValueError("key must be a non-empty bytes, str, or integer value")
        # Saved for every later ``hmac.new(self._key, ...)`` call in ``uniform``.
        self._key: bytes = key

    @staticmethod
    def _pack_bits(bits: Sequence[int]) -> bytes:
        """Serialize a 0/1 sequence as ``uint32 length || packed bits``.

        Args:
            bits: e.g. ``[1, 0, 1, 1]``.

        Returns:
            Bytes. Length is big-endian uint32, then bits packed MSB-first
            into whole bytes. Example: ``[1, 0, 1, 1]`` →
            ``b"\\x00\\x00\\x00\\x04" + b"\\xb0"`` (1011_0000).
        """
        _validate_bits(bits)
        # HMAC needs a bytes message, not a Python list of 0/1. Pack 8 bits
        # into each byte, MSB first (bit 0 of the list → bit 7 of byte 0).
        #
        # Example: bits = [1, 0, 1, 1]  (length 4)
        #   (4 + 7) // 8 = 1 byte, initially 0000_0000
        #   i=0, bit=1 → byte 0, shift 7 → 1000_0000
        #   i=1, bit=0 → byte 0, shift 6 → unchanged
        #   i=2, bit=1 → byte 0, shift 5 → 1010_0000
        #   i=3, bit=1 → byte 0, shift 4 → 1011_0000 = 0xB0
        packed = bytearray((len(bits) + 7) // 8)  # enough whole bytes; leftover bits stay 0
        for i, bit in enumerate(bits):
            packed[i // 8] |= bit << (7 - i % 8)  # i//8 = which byte; 7-i%8 = bit slot in that byte
        # Prefix with the bit-count as a 4-byte big-endian unsigned int so
        # [1, 0, 1, 1] and [1, 0, 1, 1, 0, 0, 0, 0] cannot pack the same way:
        #   struct.pack(">I", 4) == b"\x00\x00\x00\x04"
        # Full result: b"\x00\x00\x00\x04" + b"\xb0"
        return struct.pack(">I", len(bits)) + bytes(packed)

    def uniform(self, initial_bits: Sequence[int], context_bits: Sequence[int]) -> float:
        """Return a deterministic Uniform(0, 1) sample ``y``.

        Args:
            initial_bits: Random-initialization prefix ``R``, ``list[int]`` of
                0/1. Example: first ``n_star`` encoded bits, ``[1, 0, ..., 1]``.
            context_bits: Binary n-gram ``S_{i,h}``, the previous
                ``context_width`` bits. Example with h=4: ``[0, 1, 1, 0]``.

        Returns:
            ``float`` in ``(0, 1)``. Same inputs always yield the same ``y``.
            Changing either bit-string changes ``y``. The midpoint map
            ``(k + 0.5) / 2**64`` avoids returning exactly 0 or 1.
        """
        message = self.domain + self._pack_bits(initial_bits) + self._pack_bits(context_bits)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()
        return self._digest_to_unit(digest)

    @staticmethod
    def _pack_u32s(values: Sequence[int]) -> bytes:
        """Serialize integers as ``uint32 length || uint32 values`` (big-endian).

        Args:
            values: Token IDs or other non-negative ints that fit in 32 bits.
                Example: ``[11, 7, 3]``.

        Returns:
            Bytes. Example: three IDs → 4-byte count ``3`` plus 12 bytes of IDs.
        """
        packed = bytearray(struct.pack(">I", len(values)))
        for value in values:
            packed.extend(struct.pack(">I", int(value) & 0xFFFFFFFF))
        return bytes(packed)

    @staticmethod
    def _digest_to_unit(digest: bytes) -> float:
        """Map an HMAC digest to a float in ``(0, 1)`` via the 64-bit midpoint."""
        integer = int.from_bytes(digest[:8], "big")
        return (integer + 0.5) / 2**64

    def uniform_tokens(
        self,
        tokens: Sequence[int],
        extra_bits: Sequence[int] | None = None,
        *,
        domain: bytes | None = None,
    ) -> float:
        """Uniform(0, 1) sample from a token n-gram, optionally plus bit context.

        Args:
            tokens: Token IDs, ``Sequence[int]``. Example: last ``h=5`` IDs
                ``[11, 7, 3, 9, 2]``.
            extra_bits: Optional 0/1 sequence packed the same way as
                ``_pack_bits``. Example: ``[1, 0]`` for the bit index inside
                the current token, or empty.
            domain: HMAC domain separator, ``bytes``. Default
                ``domain_tokens``. CABS uses ``domain_cabs_pos``.

        Returns:
            ``float`` in ``(0, 1)``.
        """
        tag = domain if domain is not None else self.domain_tokens
        message = tag + self._pack_u32s(tokens)
        if extra_bits is not None:
            message += self._pack_bits(extra_bits)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()
        return self._digest_to_unit(digest)

    def integer_hash(self, tokens: Sequence[int], *, domain: bytes | None = None) -> int:
        """Non-negative integer hash of a token list (CABS ``Hash(Q)``).

        Args:
            tokens: Window ``Q``, ``Sequence[int]``. May be empty.
            domain: Default ``domain_cabs_frame``.

        Returns:
            ``int`` from the first 8 digest bytes. Algorithm 1 cuts a frame
            when this value ``mod 2**f == 0``.
        """
        tag = domain if domain is not None else self.domain_cabs_frame
        digest = hmac.new(self._key, tag + self._pack_u32s(tokens), hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big")


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

    Think of the vocabulary as a binary tree. At ``bit_index`` we have already
    fixed the high bits to ``prefix``, and we want the probability that the
    next bit is 1 among the remaining probability mass.

    Args:
        probabilities: 1-D vocabulary distribution. Types: ``list[float]`` or
            ``np.ndarray`` of shape ``(vocab_size,)``. Example for 4 tokens:
            ``[0.1, 0.2, 0.3, 0.4]`` (need not already sum to 1; it is
            renormalized). Extra zero-mass IDs are allowed, e.g. 5-token
            ``[0.1, 0.2, 0.3, 0.4, 0.0]`` with ``bit_length=3``.
        prefix: ``int`` whose binary form is the bits already chosen (MSB
            first). At ``bit_index=0``, prefix is always ``0``. After sampling
            a 1 then a 0, prefix is ``2`` (binary ``10``) at ``bit_index=2``.
        bit_index: ``int`` in ``[0, bit_length)``. ``0`` is the most
            significant bit of the token ID.
        bit_length: Total bits per token. Default
            ``ceil(log2(vocab_size))``. Example: vocab 5 → 3 bits.

    Returns:
        ``float`` in ``[0, 1]``. Example with
        ``probs = [0.1, 0.2, 0.3, 0.4, 0.0]`` (width 3):
        ``conditional_bit_probability(probs, prefix=0, bit_index=0) == 0.0``
        because only token 4 (binary ``100``) has MSB 1 and it has mass 0.
    """

    probs = np.asarray(probabilities, dtype=np.float64)  # shape (V,), V = vocab size
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

    # Leaves under prefix+0 and prefix+1 occupy 2^(remaining) consecutive IDs.
    remaining = width - bit_index - 1
    zero_start = (prefix << 1) << remaining  # first token ID with next bit 0
    one_start = ((prefix << 1) | 1) << remaining  # first token ID with next bit 1
    span = 1 << remaining
    zero_mass = probs[zero_start : min(zero_start + span, probs.size)].sum()
    one_mass = probs[one_start : min(one_start + span, probs.size)].sum()
    total = zero_mass + one_mass
    return float(one_mass / total) if total > 0 else 0.0


def token_ids_to_bits(token_ids: Sequence[int], bit_length: int) -> list[int]:
    """Expand token IDs into a flat MSB-first bit string.

    Args:
        token_ids: Sequence of integers, each in ``[0, 2**bit_length)``.
            Example: ``[0, 1, 7, 4]``.
        bit_length: Bits per token, positive ``int``. Example: ``3``.

    Returns:
        ``list[int]`` of 0/1. Example:
        ``token_ids_to_bits([0, 1, 7, 4], 3)``
        → ``[0, 0, 0,  0, 0, 1,  1, 1, 1,  1, 0, 0]``.
    """
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    result: list[int] = []
    for token_id in token_ids:
        if not 0 <= token_id < 2**bit_length:
            raise ValueError(f"token id {token_id} does not fit in {bit_length} bits")
        result.extend(int(bit) for bit in f"{token_id:0{bit_length}b}")
    return result


def bits_to_token_ids(bits: Sequence[int], bit_length: int) -> list[int]:
    """Inverse of ``token_ids_to_bits``: group bits into token IDs.

    Args:
        bits: Flat 0/1 sequence whose length is a multiple of ``bit_length``.
            Example: ``[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]``.
        bit_length: Bits per token. Example: ``3``.

    Returns:
        ``list[int]`` token IDs. Example: ``[0, 1, 7, 4]``.
    """
    _validate_bits(bits)
    if bit_length <= 0 or len(bits) % bit_length:
        raise ValueError("bit sequence length must be divisible by bit_length")
    return [
        int("".join(map(str, bits[i : i + bit_length])), 2) for i in range(0, len(bits), bit_length)
    ]


def in_shifted_interval(y: float, probability_one: float, delta_m: float) -> bool:
    """Membership in A_2(M), Equations (21)--(22), on the unit circle.

    The interval of length ``probability_one`` starts at ``delta_m`` and wraps
    around 0 if needed. Bit 1 is emitted iff ``y`` lands in that interval.

    Args:
        y: PRF sample, ``float`` in ``[0, 1)``. Example: ``0.95``.
        probability_one: P(bit = 1), ``float`` in ``[0, 1]``. Example: ``0.3``.
        delta_m: Message shift, ``float`` (only the fractional part matters).
            Example: payload 5, m=3 → ``0.625``. Wrap example: ``0.9``.

    Returns:
        ``bool``. With ``(y=0.95, p=0.3, delta=0.9)`` the interval is
        ``[0.9, 1.0) U [0.0, 0.2)``, so ``0.95`` and ``0.05`` are True and
        ``0.5`` is False.
    """

    if not 0.0 <= y < 1.0 or not 0.0 <= probability_one <= 1.0:
        raise ValueError("y and probability_one must lie in [0, 1]")
    start = delta_m % 1.0  # e.g. 0.9
    end = start + probability_one  # e.g. 1.2, which wraps
    return start <= y < end if end <= 1.0 else y >= start or y < end - 1.0


def disc_score(bit: int, y: float, delta_m: float) -> float:
    """DISC score from Equation (24).

    For bit 1 the circular distance is ``(y - delta_m) mod 1``; for bit 0 it
    is ``(delta_m - y) mod 1``. The score is ``-log(distance)``. Watermarked
    bits tend to land near ``delta_m`` on the matching side, so distances are
    small and scores are large.

    Args:
        bit: ``0`` or ``1``. Example: ``1``.
        y: PRF sample, ``float`` in ``[0, 1)``. Example: ``0.2``.
        delta_m: Shift, ``float``. Example: ``0.3``.

    Returns:
        ``float`` (nats). Examples:
        ``disc_score(1, 0.2, 0.3) == -log(0.9)``;
        ``disc_score(0, 0.2, 0.3) == -log(0.1)``.
    """

    if bit not in (0, 1) or not 0.0 <= y < 1.0:
        raise ValueError("bit must be binary and y must lie in [0, 1)")
    distance = ((y - delta_m) % 1.0) if bit else ((delta_m - y) % 1.0)
    # PRF midpoint output makes zero probability negligible; guard exact test inputs.
    return -math.log(max(distance, np.finfo(np.float64).tiny))


class DiscEncoder:
    """Stateful DISC encoder with optional CABS multi-position scheduling.

    Default arguments match Algorithm 3 (single position, ``prefix_bit_ngram``).
    Set ``n_positions > 1`` to split the payload into ``H`` symbols and assign
    each real token to a position with CABS; that token is then DISC-watermarked
    with the symbol at that position.

    Args:
        key: PRF secret. ``str`` / ``bytes`` / ``int``, e.g. ``"secret"``.
        payload: Integer message. With one position, in
            ``[0, 2**payload_bits)``. With ``H`` positions, in
            ``[0, 2**(payload_bits * H))``. Example: ``H=2``, ``m=3``,
            payload ``27`` → symbols ``[3, 3]``.
        payload_bits: Bits per position ``m``. Example: ``3``.
        n_positions: ``H``. Default ``1`` (plain DISC).
        context_mode: ``ContextMode``. Default ``"prefix_bit_ngram"``.
        context_width: n-gram length ``h``. Bits for bit modes, tokens for
            token modes. Default ``16``.
        use_cabs: Enable CABS. Default ``True`` iff ``n_positions > 1``.
        cabs_config: Optional ``CabsConfig``. Paper defaults if omitted.
        entropy_threshold: Random-init stop in nats. Ignored for modes
            without a prefix. Default ``5.0``.
        seed: RNG seed for unwatermarked / random-init bits.
        message_mapping: ``"direct"`` or ``"gray"`` (applied per symbol).
    """

    def __init__(
        self,
        key: bytes | str | int,
        payload: int,
        payload_bits: int,
        *,
        n_positions: int = 1,
        context_mode: ContextMode = "prefix_bit_ngram",
        entropy_threshold: float = 5.0,
        context_width: int = 16,
        seed: int | None = None,
        message_mapping: MessageMapping = "direct",
        use_cabs: bool | None = None,
        cabs_config: CabsConfig | None = None,
    ):
        if n_positions < 1 or payload_bits < 1:
            raise ValueError("n_positions and payload_bits must be positive")
        total_bits = payload_bits * n_positions
        if not 0 <= payload < 2**total_bits:
            raise ValueError("payload must be in [0, 2**(payload_bits * n_positions))")
        if entropy_threshold < 0 or context_width < 1:
            raise ValueError("entropy_threshold must be non-negative and context_width positive")
        if context_mode not in (
            "prefix_bit_ngram",
            "bit_ngram",
            "token_ngram",
            "prefix_token_ngram",
        ):
            raise ValueError("unknown context_mode")
        self.prf = HmacPrf(key)
        self.payload = payload
        self.payload_bits = payload_bits
        self.n_positions = n_positions
        self.context_mode: ContextMode = context_mode
        self.message_mapping: MessageMapping = message_mapping
        self.symbols = split_payload(payload, n_positions, payload_bits)
        self.mapped_symbols = [_map_payload(symbol, message_mapping) for symbol in self.symbols]
        self.position_deltas = [mapped / 2**payload_bits for mapped in self.mapped_symbols]
        # Backward-compatible aliases for the single-position (H=1) case.
        self.mapped_payload = self.mapped_symbols[0]
        self.delta_m = self.position_deltas[0]
        self.entropy_threshold = entropy_threshold
        self.context_width = context_width
        self.rng = random.Random(seed)
        self.bits: list[int] = []
        self.tokens: list[int] = []
        self.token_positions: list[int | None] = []
        self.empirical_entropy = 0.0
        self.n_star: int | None = None if context_mode.startswith("prefix") else 0
        self.n_star_tokens: int | None = None if context_mode == "prefix_token_ngram" else 0
        self._in_token = False
        self._token_width = 1
        self._active_delta = self.delta_m
        self.use_cabs = n_positions > 1 if use_cabs is None else use_cabs
        self.cabs: CabsScheduler | None = None
        if self.use_cabs:
            cabs_h = context_width if context_mode in ("token_ngram", "prefix_token_ngram") else 1
            self.cabs = CabsScheduler(self.prf, n_positions, cabs_h, cabs_config)

    @property
    def random_initialization(self) -> list[int]:
        """Return the random prefix ``R`` as ``list[int]`` of 0/1.

        If encoding is still initializing, this is all bits so far.
        Example after finish: ``encoder.bits[:encoder.n_star]``, e.g. 18 bits.
        """
        end = len(self.bits) if self.n_star is None else self.n_star
        return self.bits[:end]

    def _needs_random_init(self) -> bool:
        if self.context_mode == "prefix_bit_ngram":
            return self.n_star is None and (
                self.empirical_entropy < self.entropy_threshold or len(self.bits) < self.context_width
            )
        if self.context_mode == "prefix_token_ngram":
            return self.n_star_tokens is None and (
                self.empirical_entropy < self.entropy_threshold or len(self.tokens) < self.context_width
            )
        return False

    def _sample_unwatermarked(self, probability_one: float) -> int:
        bit = int(self.rng.random() < probability_one)
        chosen_probability = probability_one if bit else 1.0 - probability_one
        self.empirical_entropy += -math.log(max(chosen_probability, np.finfo(float).tiny))
        self.bits.append(bit)
        return bit

    def encode_bit(self, probability_one: float, *, watermark: bool = True, delta_m: float | None = None) -> int:
        """Sample the next binary token given P(bit = 1).

        During random init, or when ``watermark=False``, samples Bernoulli.
        Otherwise uses the shifted-interval encoder (Algorithm 3) with
        ``delta_m`` (default: the active position's shift).

        Args:
            probability_one: ``float`` in ``[0, 1]``. Example: ``0.7``.
            watermark: ``bool``. False forces an unwatermarked draw.
            delta_m: Optional shift. Example: ``0.625`` for symbol 5 with m=3.

        Returns:
            ``int`` ``0`` or ``1``. Also appends that bit to ``self.bits``.
        """
        if not 0.0 <= probability_one <= 1.0:
            raise ValueError("probability_one must lie in [0, 1]")
        cabs_position: int | None = None
        if not self._in_token and self.cabs is not None:
            cabs_position = self.cabs.propose(self.tokens)
            if cabs_position is None:
                watermark = False
            elif delta_m is None:
                delta_m = self.position_deltas[cabs_position]

        unwatermarked = self._needs_random_init() or not watermark
        if not unwatermarked and self.context_mode == "prefix_bit_ngram" and self.n_star is None:
            self.n_star = len(self.bits)
        if (
            not unwatermarked
            and self.context_mode in ("bit_ngram", "token_ngram")
            and len(self.bits) < self.context_width
        ):
            unwatermarked = True

        if unwatermarked:
            bit = self._sample_unwatermarked(probability_one)
            if (
                self.context_mode == "prefix_bit_ngram"
                and self.n_star is None
                and self.empirical_entropy >= self.entropy_threshold
                and len(self.bits) >= self.context_width
            ):
                self.n_star = len(self.bits)
        else:
            shift = self.delta_m if delta_m is None else delta_m
            token_index = len(self.tokens)
            bit_in_token = len(self.bits) % max(self._token_width, 1) if self._in_token else 0
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=self.bits,
                bit_index=len(self.bits),
                context_width=self.context_width,
                n_star=0 if self.n_star is None else self.n_star,
                tokens=self.tokens,
                token_index=token_index,
                bit_in_token=bit_in_token,
                n_star_tokens=0 if self.n_star_tokens is None else self.n_star_tokens,
            )
            bit = int(in_shifted_interval(y, probability_one, shift))
            self.bits.append(bit)

        if not self._in_token and self.cabs is not None:
            self.tokens.append(bit)
            self.token_positions.append(cabs_position)
            self.cabs.commit(bit)
        return bit

    def encode_token(self, probabilities: Sequence[float] | np.ndarray) -> int:
        """Sample one vocabulary token by walking the binarized LM tree.

        With CABS, the token is assigned to a position first; all of its bits
        use that position's DISC shift. Ineligible or warm-up tokens are
        sampled without a watermark.

        Args:
            probabilities: 1-D distribution over token IDs. Example:
                ``np.array([0.05, 0.15, 0.2, 0.25, 0.35])`` for 5 tokens.

        Returns:
            Token ID as ``int`` in ``[0, vocab_size)``. Example: ``3``.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        width = math.ceil(math.log2(probs.size))
        self._token_width = width
        position: int | None = 0
        watermark = True
        delta = self.delta_m
        if self.cabs is not None:
            position = self.cabs.propose(self.tokens)
            watermark = position is not None
            if position is not None:
                delta = self.position_deltas[position]
        elif self._needs_random_init():
            watermark = False

        self._in_token = True
        prefix = 0
        try:
            for bit_index in range(width):
                p_one = conditional_bit_probability(probs, prefix, bit_index, width)
                prefix = (prefix << 1) | self.encode_bit(
                    p_one, watermark=watermark, delta_m=delta
                )
        finally:
            self._in_token = False
        if prefix >= probs.size:
            raise RuntimeError("binarized sampling reached a zero-probability token")
        self.tokens.append(prefix)
        self.token_positions.append(position)
        if self.cabs is not None:
            self.cabs.commit(prefix)
        if (
            self.context_mode == "prefix_token_ngram"
            and self.n_star_tokens is None
            and self.empirical_entropy >= self.entropy_threshold
            and len(self.tokens) >= self.context_width
        ):
            self.n_star_tokens = len(self.tokens)
            self.n_star = len(self.bits)
        return prefix


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of ``DiscDetector.detect``.

    Attributes:
        detected: ``bool``. True iff ``global_p_value <= fpr``.
        payload: Recovered message ``int`` if detected, else ``None``.
            Example: ``5``. Already Gray-decoded when mapping is ``"gray"``.
        n_star: Best-scoring random-prefix length ``int``, or ``None`` if
            the sequence was too short. Example: ``18``.
        local_p_value: Erlang-tail p-value ``float`` in ``[0, 1]`` for the
            best (n_star, payload) pair, before multiple-testing correction.
            Example: ``1.2e-8``.
        global_p_value: Union-bound p-value ``float`` in ``[0, 1]`` from
            Equation (26). Compared against ``fpr``. Example: ``0.003``.
        score: Sum of ``disc_score`` values, ``float``. Larger is more
            watermark-like. Example: ``42.7``.
        scored_bits: Number of (context, bit) pairs that entered the score,
            ``int``. Example: ``320``. Duplicate n-grams may be skipped.
        symbols: Decoded per-position symbols ``tuple[int, ...]`` if detected.
            Example: ``(3, 3)`` for ``H=2``. ``None`` when not detected.
        position_p_values: Union-corrected p-value per position, ``tuple[float, ...]``.
            Mixed by Fisher into ``global_p_value`` when ``H > 1``.
    """

    detected: bool
    payload: int | None
    n_star: int | None
    local_p_value: float
    global_p_value: float
    score: float
    scored_bits: int
    symbols: tuple[int, ...] | None = None
    position_p_values: tuple[float, ...] = ()


class DiscDetector:
    """DISC detector (Algorithm 4) with optional CABS multi-position mixing.

    Single-position default: exhaustive search over prefix lengths and
    symbols, Erlang local p-value, union correction (Equation 26).

    Multi-position / CABS: replay CABS, DISC-score each position's tokens
    independently, union-correct each position by ``|M| = 2**m``, then mix
    those q-values with Fisher's method so the overall test still uses the
    same ``fpr``.

    Args:
        key: Same secret as the encoder.
        payload_bits: Bits per position ``m``.
        n_positions: ``H``. Default ``1``.
        context_mode: Must match the encoder.
        context_width: ``h``.
        fpr: Overall false-positive target. Default ``0.01``.
        use_cabs: Default ``True`` iff ``n_positions > 1``.
        cabs_config: Optional ``CabsConfig``.
    """

    def __init__(
        self,
        key: bytes | str | int,
        payload_bits: int,
        *,
        n_positions: int = 1,
        context_mode: ContextMode = "prefix_bit_ngram",
        context_width: int = 16,
        fpr: float = 0.01,
        deduplicate_ngrams: bool = True,
        message_mapping: MessageMapping = "direct",
        use_cabs: bool | None = None,
        cabs_config: CabsConfig | None = None,
    ):
        if payload_bits < 1 or context_width < 1 or n_positions < 1:
            raise ValueError("payload_bits, context_width, and n_positions must be positive")
        if not 0 < fpr < 1:
            raise ValueError("fpr must lie strictly between 0 and 1")
        if context_mode not in (
            "prefix_bit_ngram",
            "bit_ngram",
            "token_ngram",
            "prefix_token_ngram",
        ):
            raise ValueError("unknown context_mode")
        self.prf = HmacPrf(key)
        self.payload_bits = payload_bits
        self.n_positions = n_positions
        self.context_mode: ContextMode = context_mode
        self.message_count = 2**payload_bits
        self.context_width = context_width
        self.fpr = fpr
        self.deduplicate_ngrams = deduplicate_ngrams
        _map_payload(0, message_mapping)
        self.message_mapping: MessageMapping = message_mapping
        self.use_cabs = n_positions > 1 if use_cabs is None else use_cabs
        self.cabs_config = cabs_config
        self._cabs_h = context_width if context_mode in ("token_ngram", "prefix_token_ngram") else 1

    def _score(self, bits: Sequence[int], n_star: int, payload: int) -> tuple[float, int]:
        """Sum Equation (24) scores for one hypothesized (n_star, payload).

        Args:
            bits: Full 0/1 sequence, ``list[int]``. Example length 340.
            n_star: Hypothesized prefix length, ``int``. Example: ``18``.
            payload: Hypothesized *mapped* interval index, ``int`` in
                ``[0, 2**m)``. Example: ``5``.

        Returns:
            ``(score, count)`` where ``score`` is ``float`` (sum of nats)
            and ``count`` is ``int`` scored positions. Example: ``(42.7, 320)``.
        """
        delta_m = payload / self.message_count
        score = 0.0
        count = 0
        seen: set[tuple[int, ...]] = set()
        start = n_star if self.context_mode == "prefix_bit_ngram" else max(n_star, self.context_width)
        for index in range(start, len(bits)):
            context = bits[index - self.context_width : index]
            ngram = tuple(context) + (bits[index],)
            if self.deduplicate_ngrams and ngram in seen:
                continue
            seen.add(ngram)
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=bits,
                bit_index=index,
                context_width=self.context_width,
                n_star=n_star if self.context_mode == "prefix_bit_ngram" else 0,
            )
            score += disc_score(bits[index], y, delta_m)
            count += 1
        return score, count

    def _score_indices(
        self,
        bits: Sequence[int],
        bit_indices: Sequence[int],
        mapped_symbol: int,
        *,
        n_star: int,
        tokens: Sequence[int],
        bit_length: int,
        n_star_tokens: int,
    ) -> tuple[float, int]:
        """DISC score a subset of bit indices with one position's symbol."""
        delta_m = mapped_symbol / self.message_count
        score = 0.0
        count = 0
        seen: set[tuple[int, ...]] = set()
        for bit_index in bit_indices:
            if bit_index < self.context_width and self.context_mode in (
                "prefix_bit_ngram",
                "bit_ngram",
            ):
                continue
            token_index = bit_index // bit_length if bit_length else 0
            bit_in_token = bit_index % bit_length if bit_length else 0
            if self.context_mode in ("token_ngram", "prefix_token_ngram"):
                if token_index < self.context_width:
                    continue
                ngram = tuple(tokens[token_index - self.context_width : token_index]) + (
                    bits[bit_index],
                )
            else:
                ngram = tuple(bits[bit_index - self.context_width : bit_index]) + (bits[bit_index],)
            if self.deduplicate_ngrams and ngram in seen:
                continue
            seen.add(ngram)
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=bits,
                bit_index=bit_index,
                context_width=self.context_width,
                n_star=n_star,
                tokens=tokens,
                token_index=token_index,
                bit_in_token=bit_in_token,
                n_star_tokens=n_star_tokens,
            )
            score += disc_score(bits[bit_index], y, delta_m)
            count += 1
        return score, count

    def _detect_positions(
        self,
        bits: list[int],
        token_ids: Sequence[int],
        bit_length: int,
        n_star: int,
        n_star_tokens: int,
    ) -> DetectionResult:
        """Per-position DISC tests mixed by Fisher combination of q-values."""
        if self.use_cabs:
            scheduler = CabsScheduler(
                self.prf, self.n_positions, self._cabs_h, self.cabs_config
            )
            assignments = scheduler.assign_sequence(token_ids)
        else:
            assignments = [index % self.n_positions for index in range(len(token_ids))]

        grouped: list[list[int]] = [[] for _ in range(self.n_positions)]
        for token_index, position in enumerate(assignments):
            if position is None:
                continue
            start = token_index * bit_length
            grouped[position].extend(range(start, start + bit_length))

        symbols: list[int] = []
        position_p: list[float] = []
        total_score = 0.0
        total_count = 0
        local_p_values: list[float] = []
        for position in range(self.n_positions):
            if not grouped[position]:
                symbols.append(0)
                continue
            best_local = 1.0
            best_mapped = 0
            best_score = 0.0
            best_count = 0
            for mapped in range(self.message_count):
                score, count = self._score_indices(
                    bits,
                    grouped[position],
                    mapped,
                    n_star=n_star,
                    tokens=token_ids,
                    bit_length=bit_length,
                    n_star_tokens=n_star_tokens,
                )
                local_p = float(gammaincc(count, score)) if count else 1.0
                if local_p < best_local:
                    best_local, best_mapped, best_score, best_count = (
                        local_p,
                        mapped,
                        score,
                        count,
                    )
            symbols.append(_unmap_payload(best_mapped, self.message_mapping))
            position_p.append(min(1.0, self.message_count * best_local))
            local_p_values.append(best_local)
            total_score += best_score
            total_count += best_count

        global_p = combine_p_values(position_p)
        detected = global_p <= self.fpr
        payload = join_payload(symbols, self.payload_bits) if detected else None
        local_p = min(local_p_values) if local_p_values else 1.0
        return DetectionResult(
            detected,
            payload,
            n_star,
            local_p,
            global_p,
            total_score,
            total_count,
            tuple(symbols) if detected else None,
            tuple(position_p),
        )

    def detect(
        self,
        bits: Sequence[int],
        *,
        n_star_candidates: Iterable[int] | None = None,
        token_ids: Sequence[int] | None = None,
        bit_length: int | None = None,
    ) -> DetectionResult:
        """Search payloads and prefix lengths; return the corrected test.

        Args:
            bits: Observed binary stream.
            n_star_candidates: Optional prefix lengths for prefix modes.
            token_ids: Real token IDs, required for CABS / token n-grams.
                If omitted with CABS, each bit is treated as a 0/1 token.
            bit_length: Bits per token. Default ``len(bits) // len(token_ids)``.

        Returns:
            ``DetectionResult``.
        """
        bits = list(bits)
        _validate_bits(bits)
        if self.use_cabs or self.n_positions > 1 or self.context_mode in (
            "token_ngram",
            "prefix_token_ngram",
        ):
            if token_ids is None:
                token_ids = bits
                bit_length = 1
            else:
                token_ids = list(token_ids)
                bit_length = bit_length or (len(bits) // max(len(token_ids), 1))
            n_star = 0
            n_star_tokens = 0
            if n_star_candidates is not None:
                candidates = [c for c in n_star_candidates if c is not None]
                n_star = candidates[0] if candidates else 0
            return self._detect_positions(bits, token_ids, bit_length, n_star, n_star_tokens)

        if len(bits) <= self.context_width:
            return DetectionResult(False, None, None, 1.0, 1.0, 0.0, 0)
        if self.context_mode == "bit_ngram":
            n_star_candidates = [self.context_width] if n_star_candidates is None else n_star_candidates
        candidates = (
            list(n_star_candidates)
            if n_star_candidates is not None
            else range(self.context_width, len(bits))
        )
        best: tuple[float, float, int, int | None, int | None] = (1.0, 0.0, 0, None, None)
        for n_star in candidates:
            if not self.context_width <= n_star < len(bits):
                continue
            for payload in range(self.message_count):
                score, count = self._score(bits, n_star, payload)
                local_p = float(gammaincc(count, score)) if count else 1.0
                if local_p < best[0]:
                    best = (local_p, score, count, n_star, payload)

        local_p, score, count, n_star, mapped_payload = best
        per_start = min(1.0, self.message_count * local_p)
        n_start_tests = (
            1 if self.context_mode == "bit_ngram" else (len(bits) - self.context_width)
        )
        global_p = (
            -math.expm1(n_start_tests * math.log1p(-per_start))
            if per_start < 1.0
            else 1.0
        )
        detected = global_p <= self.fpr
        payload: int | None = None
        if detected and mapped_payload is not None:
            payload = _unmap_payload(mapped_payload, self.message_mapping)
        return DetectionResult(
            detected,
            payload,
            n_star,
            local_p,
            global_p,
            score,
            count,
        )
