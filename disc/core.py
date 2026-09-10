"""Paper-faithful, model-independent implementation of DISC.

The implementation follows Algorithms 3 and 4 and Equations (20)--(26) of
Kordi Boroujeny et al., "Multi-Bit Distortion-Free Watermarking for Large
Language Models" (arXiv:2402.16578v1).

All logarithms are natural logarithms, as in the paper. Positions in this
module are zero based. ``n_star`` is how many leading bits were generated
from the LM with no watermark; those bits are the prefix ``R`` used later
as a PRF input. The decoder searches candidate ``n_star`` values.

Data the rest of the package works with
---------------------------------------
- ``bits``: ``list[int]`` of 0/1 values, e.g. ``[1, 0, 1, 1, 0]``.
- ``payload_bits`` (``m``): bits stored in one DISC position.
  Example: ``m = 4`` means each position holds a symbol in ``{0, ..., 15}``.
- ``n_positions`` (``H``): how many DISC positions the message is split into.
  Each position is watermarked with DISC. ``H`` may be 1 or greater;
  CABS is only the optional scheduler that assigns tokens to those positions.
- ``payload``: the full integer message, ``m * H`` bits long.
  Range: ``{0, ..., 2**(m * H) - 1}``.
  Example: ``m = 4``, ``H = 2`` → 8-bit payload in ``{0, ..., 255}``.
  ``payload = 27`` splits as symbols ``[1, 11]`` because ``27 = 1 * 16 + 11``.
- ``delta_m``: the unit-interval shift used by DISC at one position,
  ``symbol / 2**m``. It is computed from that position's symbol, not from
  the full payload. In the example: ``1/16 = 0.0625`` and ``11/16 = 0.6875``.
  If ``H = 1`` there is only one symbol, so ``delta_m = payload / 2**m``.
- ``probability_one`` / ``p_one``: ``float`` in ``[0, 1]``, P(next bit = 1).
- ``y``: ``float`` in ``[0, 1)``, HMAC-PRF output used as a uniform sample.
- ``n_star``: ``int``, length of the unwatermarked prefix ``R``.
  Encoder: first ``n_star`` bits are sampled from the LM, then PRF uses ``R``.
  Decoder: ``n_star`` is unknown, so each candidate start treats
  ``bits[:n_star]`` as ``R``.
    ``use_prefix=False`` (or ``context_mode`` ``bit_ngram`` / ``token_ngram``)
    means no ``R``: ``n_star = 0``, ``R = []``, and the decoder does not search
    starts. The first ``h`` tokens/bits are still generated without a watermark
    so the n-gram exists; they are not hashed as ``R``. With
    ``randomize_without_prefix=True``, a second domain-separated PRF decides
    whether each eligible token/bit is watermarked; otherwise only the DISC PRF
    is used and the no-prefix schedule is deterministic for a fixed context.
- ``context_width`` (``h``): context length in real tokens, as in the papers.
  The binary n-gram is ``S_{i,h} = W^b_{[i - h w : i - 1]}`` with
  ``w = ceil(log2 |V|)``, i.e. the last ``h * w`` bits.
  For a pure bit stream (``|V| = 2``, Bernoulli tests) ``w = 1``, so ``h``
  bits. CABS always uses the last ``h`` token IDs.
- ``watermark_mask``: optional ``list[bool]`` parallel to ``bits``. ``False``
    marks a bit intentionally sampled from the LM and excluded by the detector.

Call flow
---------
Encode (DiscEncoder.encode_token / encode_bit)::

    encode_token
      └─ encode_bit                 once per bit of the real token
           ├─ _sample_unwatermarked if use_prefix: first n_star bits are R
           │                        if not: only h tokens/bits of n-gram warm-up
           ├─ CabsScheduler.propose only if H>1 / CABS: which position this token uses
           ├─ selector PRF             optional second PRF for probabilistic R=[] mode
           └─ prf_uniform           y = F(R, context); R=[] when use_prefix=False
                ├─ HmacPrf.uniform          bit modes
                │     _pack_bits + _digest_to_unit
                └─ HmacPrf.uniform_tokens   token modes
                      _pack_u32s + _digest_to_unit

Detect (DiscDetector.detect)::

    detect
      ├─ if use_prefix: try candidate n_star; bits[:n_star] is hypothesized R
      │  if not: n_star=0, R=[] (no start search)
    ├─ selector PRF (optional) reconstructs the R=[] watermark mask
    ├─ _score / _score_indices / _detect_positions
      │     same prf_uniform → uniform / uniform_tokens as encode
      ├─ disc_score + Erlang tail
      └─ combine_p_values           only if H>1

Why the layers exist
--------------------
- DiscEncoder / DiscDetector: paper Algorithms 3 and 4 (public API).
- prf_uniform: shared entry so encode and detect slice the same R and n-gram.
- HmacPrf.uniform / uniform_tokens: the PRF F itself (HMAC → y in (0, 1)).
- _pack_bits / _pack_u32s: HMAC needs bytes, not Python lists.
- _digest_to_unit: 32-byte digest → y in (0, 1); shared by both PRF methods.
- ``domain_randomization``: separate HMAC domain for the optional no-prefix
    watermark-selection PRF; it is distinct from the DISC ``y`` domain.
- split_payload, disc_score, in_shifted_interval: math used on both sides.
Leading-underscore helpers are not a second API; they only avoid duplicating
packing and the digest-to-float map.

Reading guide
-------------
The public workflow has two entry points. ``DiscEncoder`` creates a stream and
``DiscDetector`` analyzes a stream. The encoder and detector deliberately share
the same small mathematical and PRF helpers so that the detector reconstructs
exactly the value that the encoder used for each bit.

Encoder dependency flow::

        DiscEncoder.__init__
            ├─ HmacPrf(key)                    normalize and store the secret
            ├─ split_payload(payload, m, H)     one symbol per DISC position
            └─ _map_payload(symbol, mapping)    direct or Gray-coded symbol

        DiscEncoder.encode_token(probabilities)
            ├─ conditional_bit_probability     convert the token distribution to a
            │                                    conditional probability for each bit
            ├─ CabsScheduler.propose            choose a position when CABS is active
            └─ encode_bit(p_one, delta_m)
                     ├─ _needs_random_init         decide whether prefix R is still growing
                     ├─ _needs_context_warmup      wait until the n-gram exists
                     ├─ _sample_unwatermarked      sample ordinary Bernoulli bits, or
                     ├─ prf_uniform                 select R and the context, then call
                     │    └─ HmacPrf.uniform*       serialize inputs and compute HMAC
                     └─ in_shifted_interval         turn y and delta_m into bit 0 or 1

Detector dependency flow::

        DiscDetector.detect(bits, token_ids=...)
            ├─ _validate_bits                  reject values other than 0 and 1
            ├─ _score                         single-position search over starts/M, or
            └─ _detect_positions               replay CABS and score each position
                     ├─ _score_indices             score only tokens assigned to one position
                     ├─ prf_uniform -> HmacPrf     reconstruct the encoder's PRF input
                     ├─ disc_score                  compute Equation (24)
                     ├─ gammaincc                   convert the score to an Erlang p-value
                     ├─ _unmap_payload              undo optional Gray mapping
                     └─ combine_p_values             combine positions with Fisher's method

The class decorators have intentionally modest roles. ``@property`` exposes a
derived read-only value such as ``encoder.binary_context_width`` without a
method call. ``@staticmethod`` marks a helper such as ``HmacPrf._pack_bits``
that belongs conceptually to the class but does not read ``self`` or class
state. Neither decorator changes the watermarking algorithm; they clarify how
the surrounding API is used.
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

# "direct" embeds one position's symbol M as delta = M / 2^m (paper default).
# "gray" first maps that symbol to binary-reflected Gray code, then uses it.
# Example: symbol 6 with m=3 → direct uses 6; gray uses gray_encode(6) = 5.
# With H>1 this is applied separately to each position, not to the full payload.
MessageMapping = Literal["direct", "gray"]

# How the PRF is seeded for each binary token (DISC §3.2 vs §3.3, plus token n-grams).
# h is always in real tokens; bit modes expand it to h * ceil(log2|V|) bits.
#
# R is not a random number. In the encoder the first n_star bits/tokens are
# sampled from the LM (or Bernoulli) with no watermark; that prefix is R.
# After that, and in the decoder, y = F(R, context). The decoder does not
# know n_star, so it tries several candidate lengths and treats the first
# n_star bits of the observed binary stream as R.
#   "prefix_bit_ngram"    — y = F(R, last h*w bits). Default DISC (uses R).
#   "bit_ngram"           — y = F(last h*w bits). No R (DISC without prefix).
#   "token_ngram"         — y = F(last h token IDs, bit index). No R.
#   "prefix_token_ngram"  — y = F(R as token prefix, last h token IDs, bit index).
# Set use_prefix=False on the encoder/detector to keep a prefix_* mode but
# hash R=[] and skip the decoder's n_star search (same as the no-R modes).
# Set randomize_without_prefix=True on both sides to use a second,
# domain-separated PRF to decide whether each no-prefix token/bit is watermarked.
ContextMode = Literal["prefix_bit_ngram", "bit_ngram", "token_ngram", "prefix_token_ngram"]


def _resolve_use_prefix(context_mode: ContextMode, use_prefix: bool | None) -> bool:
    """True if encode/detect should collect and search a nonempty prefix ``R``.

    ``None`` follows the context mode: prefix_* → True, otherwise False.
    ``False`` forces ``R = []`` even for prefix_* modes.

    Args:
        context_mode: ContextMode: One string: "prefix_bit_ngram", "bit_ngram",
            "token_ngram", or "prefix_token_ngram"; selects the PRF context
            representation.
        use_prefix: bool | None: Scalar prefix flag. None follows context_mode; False
            uses an empty prefix. True requires a prefix_* mode.

    Returns:
        bool: One scalar; True enables prefix collection/search and False uses R = [].
    """
    # ``None`` delegates the decision to the mode.  This keeps the convenient
    # prefix-mode default while still allowing callers to force an empty R.
    if use_prefix is None:
        return context_mode.startswith("prefix")
    # Non-prefix modes have no R field in their logical PRF input.  Rejecting
    # this mismatch early prevents encoder and detector hashes from diverging.
    if use_prefix and not context_mode.startswith("prefix"):
        raise ValueError("use_prefix=True requires prefix_bit_ngram or prefix_token_ngram")
    # Store an ordinary bool even if a bool-like object was supplied.
    return bool(use_prefix)


def _binary_cabs_settings(
    config: CabsConfig | None,
    n_positions: int,
    context_width: int,
    bits_per_token: int,
) -> tuple[int, CabsConfig]:
    """Scale CABS from real-token units to binary-token units.

    Args:
        config: Optional CABS configuration expressed in real-token units.
        n_positions: Positive number of payload positions H.
        context_width: Positive DISC context length h in real tokens.
        bits_per_token: Positive binary width w of one real token.

    Returns:
        tuple[int, CabsConfig]: The binary eligibility width ``h*w`` and a
        configuration with window/minimum/maximum frame lengths in bits.
        ``frame_bits`` becomes ``f + ceil(log2(w))`` so the hash-cut rate per
        original real token remains approximately unchanged.
    """
    base = config or CabsConfig()
    minimum = base.min_len if base.min_len is not None else n_positions
    scaled = CabsConfig(
        window_size=base.window_size * bits_per_token,
        frame_bits=base.frame_bits + math.ceil(math.log2(bits_per_token)),
        max_factor=base.max_factor * bits_per_token,
        min_len=minimum * bits_per_token,
    )
    return context_width * bits_per_token, scaled


# ---------------------------------------------------------------------------
# Payload mapping and multi-position helpers (used by both encode and detect)
# ---------------------------------------------------------------------------


def gray_encode(value: int) -> int:
    """Map a non-negative integer to its binary-reflected Gray code.

    Args:
        value: int: Nonnegative scalar integer of arbitrary bit width.

    Returns:
        int: One nonnegative Gray-coded scalar, with the same bit length as value (0
        maps to 0).
    """
    # DISC position symbols are unsigned interval indices, so signed values
    # have no valid Gray-code interpretation in this implementation.
    if value < 0:
        raise ValueError("value must be non-negative")
    # XORing the value with its one-bit right shift produces the reflected
    # Gray-code index while preserving a one-to-one mapping.
    return value ^ (value >> 1)


def gray_decode(value: int) -> int:
    """Invert a binary-reflected Gray-code integer.

    Args:
        value: int: Nonnegative scalar Gray-coded integer of arbitrary bit width.

    Returns:
        int: One decoded nonnegative scalar, with the same bit length as value.
    """
    # Apply the same unsigned-domain check as the forward transformation.
    if value < 0:
        raise ValueError("value must be non-negative")
    decoded = 0
    # XOR successive right-shifts: 5 (101) → 5 ^ 2 ^ 1 = 6.
    # Fold progressively shorter Gray-code prefixes into the decoded value;
    # one iteration is performed for every significant input bit.
    while value:
        decoded ^= value
        value >>= 1
    return decoded


def _map_payload(payload: int, mapping: MessageMapping) -> int:
    """Convert a user payload to the interval index actually embedded.

    Args:
        payload: int: One nonnegative position symbol, normally in [0, 2**payload_bits).
        mapping: MessageMapping: One string, "direct" for identity or "gray" for binary-
            reflected Gray mapping.

    Returns:
        int: One mapped position symbol in the same symbol range.
    """
    # Mapping is applied before converting the symbol to an interval shift.
    # Spell out both supported branches so a misspelled configuration cannot
    # silently select a different representation.
    if mapping == "direct":
        return payload
    if mapping == "gray":
        return gray_encode(payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _unmap_payload(mapped_payload: int, mapping: MessageMapping) -> int:
    """Invert ``_map_payload`` after the detector recovers an interval index.

    Args:
        mapped_payload: int: One nonnegative mapped position symbol, normally in [0,
            2**payload_bits).
        mapping: MessageMapping: One string, "direct" for identity or "gray" for binary-
            reflected Gray mapping.

    Returns:
        int: One user-facing position symbol after reversing the mapping.
    """
    # Detection scores the mapped index; convert it back only at the API edge.
    # Keeping the search in mapped space makes its candidate interval indices
    # exactly match those used by the encoder.
    if mapping == "direct":
        return mapped_payload
    if mapping == "gray":
        return gray_decode(mapped_payload)
    raise ValueError("message_mapping must be 'direct' or 'gray'")


def _validate_bits(bits: Sequence[int]) -> None:
    """Raise if ``bits`` is not a sequence of 0/1 integers.

    Args:
        bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
            and 1 in stream order.

    Returns:
        None: Validates the sequence without changing it; raises ValueError for a value
        other than 0 or 1.
    """
    # Keep malformed observations out of both HMAC serialization and scoring.
    # The membership test also accepts integer scalar types from libraries
    # such as NumPy while rejecting every value other than exact 0 and 1.
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


def split_payload(payload: int, n_positions: int, symbol_bits: int) -> list[int]:
    """Split an integer payload into ``H`` big-endian ``symbol_bits``-wide symbols.

    Args:
        payload: int: Scalar message in [0, 2**(n_positions * symbol_bits)).
        n_positions: int: Positive scalar H, the number of DISC positions.
        symbol_bits: int: Positive scalar m, the fixed bit width of each symbol.

    Returns:
        list[int]: Flat list of length n_positions, most-significant symbol first; each
        symbol is in [0, 2**symbol_bits).
    """
    # There must be at least one nonempty position to hold a message.
    if n_positions < 1 or symbol_bits < 1:
        raise ValueError("n_positions and symbol_bits must be positive")
    # Peel off low-order symbols, then reverse so the result is big-endian.
    mask = (1 << symbol_bits) - 1
    symbols = []
    value = payload
    # Extract exactly H low-order chunks.  Fixed-width masking preserves zero
    # symbols that an ordinary integer representation would omit.
    for _ in range(n_positions):
        symbols.append(value & mask)
        value >>= symbol_bits
    # Remaining high-order bits mean the requested H*m-bit container is too
    # small and joining the extracted symbols would lose information.
    if value:
        raise ValueError("payload does not fit in n_positions * symbol_bits")
    symbols.reverse()
    return symbols


def join_payload(symbols: Sequence[int], symbol_bits: int) -> int:
    """Inverse of ``split_payload``.

    Args:
        symbols: Sequence[int]: Flat sequence of H symbols in big-endian order, each
            expected in [0, 2**symbol_bits). H may be zero.
        symbol_bits: int: Positive scalar m, the fixed bit width of each symbol.

    Returns:
        int: One combined scalar in [0, 2**(H * symbol_bits)) for valid symbols; an
        empty sequence returns 0.
    """
    # Shift the accumulated prefix left to make room for each next symbol.
    # Input order is therefore big-endian: the first symbol becomes the most
    # significant chunk of the combined payload.
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
        p_values: Sequence[float]: Flat sequence of H scalar p-values in [0, 1],
            possibly empty. Values are clamped to [float64.tiny, 1] before logarithms.

    Returns:
        float: One combined p-value in [0, 1]; 1.0 for an empty sequence.
    """
    # Clamp values before taking logs; p=0 can arise from floating-point
    # underflow but must still produce a finite Fisher statistic.
    values = [min(1.0, max(float(p), np.finfo(np.float64).tiny)) for p in p_values]
    # An empty collection contains no evidence against the null hypothesis,
    # so its neutral combined p-value is one.
    if not values:
        return 1.0
    # Fisher's statistic follows an Erlang distribution under the null.
    score = -sum(math.log(p) for p in values)
    return float(gammaincc(len(values), score))


# ---------------------------------------------------------------------------
# PRF: prf_uniform chooses R and context; HmacPrf is F itself
# ---------------------------------------------------------------------------


def prf_uniform(
    prf: HmacPrf,
    context_mode: ContextMode,
    *,  # everything after this must be passed by name (bits=..., bit_index=..., ...)
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

    Called by ``DiscEncoder.encode_bit`` and ``DiscDetector`` scoring. This
    function only slices ``R`` and the n-gram; ``HmacPrf`` computes ``F``.

    The ``*`` in the signature makes ``bits``, ``bit_index``, ``context_width``,
    and the later parameters keyword-only. That avoids mixing up two integers
    (bit index vs h) if someone calls this positionally.

    ``R`` is not a random number. It is the unwatermarked prefix of length
    ``n_star`` (bits) or ``n_star_tokens`` (tokens). The encoder samples those
    first tokens from the LM with no watermark; after that this function
    hashes ``R`` together with the n-gram. The decoder does not know
    ``n_star``, so each candidate start passes ``bits[:n_star]`` (or
    ``tokens[:n_star_tokens]``) as ``R``.

    Prefix modes: ``y = F(R, context)``. Non-prefix modes: ``y = F(context)``.

    Args:
        prf: HmacPrf: One initialized keyed PRF object.
        context_mode: ContextMode: One string: "prefix_bit_ngram", "bit_ngram",
            "token_ngram", or "prefix_token_ngram"; selects the PRF context
            representation.
        bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
            and 1 in stream order.
        bit_index: int: Zero-based scalar index of the current bit. bits must contain
            all preceding context and prefix bits.
        context_width: int: Nonnegative scalar slice length in bits for bit modes or
            token IDs for token modes; the preceding window must exist.
        n_star: int: Nonnegative scalar prefix length in bits; the prefix is
            bits[:n_star].
        tokens: Sequence[int] | None: Flat token history or full observation of length
            N_tokens with unsigned 32-bit IDs; required in token modes.
        token_index: int | None: Zero-based current-token index, required in token
            modes; tokens must contain its preceding context and prefix.
        bit_in_token: int: Nonnegative scalar bit offset within the current token, with
            0 denoting the MSB.
        n_star_tokens: int: Nonnegative scalar prefix length in tokens; used by
            prefix_token_ngram.

    Returns:
        float: One deterministic PRF sample, mathematically in (0, 1); floating-point
        rounding can produce 1.0.
    """
    # Bit modes slice both logical inputs directly from the binary stream.
    if context_mode in ("prefix_bit_ngram", "bit_ngram"):
        # Bit modes take the previous h*w bits immediately before the bit
        # being generated or scored.
        start = bit_index - context_width
        if start < 0:
            raise ValueError("bit_index is too small for context_width")
        context = bits[start:bit_index]
        # Prefix mode: R is the leading unwatermarked bits, not a random seed.
        # Decoder: n_star is a hypothesized start, so R = bits[:n_star].
        initial = bits[:n_star] if context_mode == "prefix_bit_ngram" else []
        return prf.uniform(initial, context)
    # Token modes require explicit token boundaries; these cannot in general
    # be reconstructed from bits without the vocabulary's encoding width.
    if tokens is None or token_index is None:
        raise ValueError("token n-gram context requires tokens and token_index")
    if token_index < context_width:
        raise ValueError("token_index is too small for context_width")
    # Token modes use real token IDs for history and add the current bit's
    # position so bits within the same token receive distinct PRF inputs.
    history = tokens[token_index - context_width : token_index]
    # Serialize the bit's offset as a fixed eight-bit value.  This separates
    # PRF samples for different bits of the same current vocabulary token.
    extra = [(bit_in_token >> shift) & 1 for shift in range(7, -1, -1)]
    if context_mode == "token_ngram":
        return prf.uniform_tokens(history, extra)
    if context_mode == "prefix_token_ngram":
        # R = tokens[:n_star_tokens]: LM-sampled prefix (encoder) or a
        # hypothesized start (decoder). Concatenated with the last h IDs.
        return prf.uniform_tokens(list(tokens[:n_star_tokens]) + list(history), extra)
    # Reaching this point means a runtime string escaped static Literal checks.
    raise ValueError("context_mode must be a ContextMode literal")


class HmacPrf:
    """HMAC-SHA256 PRF with an explicit, portable input encoding.

    This is the paper's ``F``. Encode and detect never call it directly;
    they go through ``prf_uniform`` (or CABS, which uses ``uniform_tokens``
    / ``integer_hash``).

    Maps a secret key plus two bit-strings (the unwatermarked prefix ``R``
    and n-gram context ``S``) to a uniform sample ``y`` in ``(0, 1)``.
    ``domain_randomization`` is a separate domain used only to decide whether
    an ``R=[]`` token/bit is watermarked; it is not used to generate ``y``.

    Args / attributes:
        key: Secret. Accepted types:
            - ``str``, e.g. ``"secret"`` (UTF-8 encoded);
            - ``bytes``, e.g. ``b"secret"``;
            - non-negative ``int``, e.g. ``42`` (big-endian bytes).

    Example::

        prf = HmacPrf("secret")
        y = prf.uniform(initial_bits=[1, 0, 1], context_bits=[0, 1, 1, 0])
        # y is float, e.g. 0.137..., always in (0, 1), never exactly 0 or 1

        Responsibilities:
                ``HmacPrf`` is the cryptographic layer only. It does not decide when
                to watermark, choose a context window, interpret a payload, or score
                a sequence. ``prf_uniform`` prepares those logical inputs; this class
                serializes them, computes HMAC-SHA256, and maps the digest to ``y``.

        Method flow::

                uniform(initial_bits, context_bits)
                    -> _pack_bits(R), _pack_bits(S)
                    -> domain + serialized inputs
                    -> HMAC-SHA256
                    -> _digest_to_unit

                uniform_tokens(tokens, extra_bits)
                    -> _pack_u32s(tokens), optionally _pack_bits(extra_bits)
                    -> token domain + serialized inputs
                    -> HMAC-SHA256
                    -> _digest_to_unit

                integer_hash(tokens)
                    -> _pack_u32s(tokens)
                    -> CABS frame domain + HMAC-SHA256
                    -> first eight digest bytes as an integer

        ``_pack_bits`` and ``_pack_u32s`` are ``@staticmethod`` helpers because
        their output depends only on their arguments. They can therefore be read
        and tested as serialization functions independently of a particular PRF
        instance, although callers normally reach them through this class.
    """

    # Domain separators so other HMAC uses of the same key cannot collide.
    domain = b"DISC-v1\x00"
    domain_tokens = b"DISC-v1-tok\x00"
    domain_randomization = b"DISC-v1-randomize\x00"
    domain_cabs_pos = b"DISC-v1-cabs-pos\x00"
    domain_cabs_frame = b"DISC-v1-cabs-frame\x00"

    def __init__(self, key: bytes | str | int):
        """Normalize and store the secret used by all HMAC operations.

        The public PRF methods accept a string, bytes, or non-negative integer
        key. Strings and integers are converted to bytes once here so later
        calls to ``uniform``, ``uniform_tokens``, and ``integer_hash`` can use
        the same normalized key.

        Raises:
            ValueError: If the key is empty, negative, or has an unsupported
                type.

        Args:
            key: bytes | str | int: Nonempty byte string or UTF-8 text of arbitrary length,
                or a nonnegative scalar integer encoded as at least one big-endian byte.

        Returns:
            None: Initializes this instance and stores its normalized secret bytes.
        """
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
            bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
                and 1 in stream order.

        Returns:
            bytes: 4 + ceil(N_bits / 8) bytes: unsigned 32-bit big-endian bit count followed
            by MSB-first packed bits, with zero padding in the last byte. N_bits must fit in
            32 bits.
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

    def uniform(
        self,
        initial_bits: Sequence[int],
        context_bits: Sequence[int],
        *,
        domain: bytes | None = None,
    ) -> float:
        """Return a deterministic Uniform(0, 1) sample ``y``.

        Called from ``prf_uniform`` for bit modes. This is ``F(R, S)``;
        packing and the digest map are ``_pack_bits`` and ``_digest_to_unit``.

        Args:
            initial_bits: Sequence[int]: Flat 0/1 prefix R of length N_prefix, possibly
                zero.
            context_bits: Sequence[int]: Flat 0/1 context of length N_context, normally h *
                bits_per_token; both sequence lengths must fit in 32 bits.
            domain: bytes | None: Byte-string domain separator of arbitrary length; None
                uses this method’s default domain.

        Returns:
            float: One deterministic sample from the 64-bit midpoint map; mathematically in
            (0, 1), though float rounding can yield 1.0.
        """
        # Include lengths and domain separation in the HMAC message so the
        # prefix and context cannot be ambiguously concatenated.
        # Use the ordinary DISC domain unless the caller explicitly requests a
        # disjoint logical use such as watermark-selection randomization.
        tag = self.domain if domain is None else domain
        # Each bit sequence carries its own length prefix, so concatenating R
        # and S is unambiguous even when either ends in zero-padding bits.
        message = tag + self._pack_bits(initial_bits) + self._pack_bits(context_bits)
        # HMAC supplies a stable keyed digest; the conversion below uses only
        # its first 64 bits to obtain a reproducible unit-interval sample.
        digest = hmac.new(self._key, message, hashlib.sha256).digest()
        return self._digest_to_unit(digest)

    @staticmethod
    def _pack_u32s(values: Sequence[int]) -> bytes:
        """Serialize integers as ``uint32 length || uint32 values`` (big-endian).

        Used by ``uniform_tokens`` and ``integer_hash``. HMAC cannot hash a
        Python list, so token IDs become: 4-byte count, then 4 bytes per ID.
        ``>I`` means big-endian unsigned 32-bit int.

        Args:
            values: Sequence[int]: Flat sequence of N integers, N < 2**32. Each value is
                converted to int and masked to its low 32 bits.

        Returns:
            bytes: Exactly 4 + 4*N bytes: unsigned 32-bit big-endian count followed by N
            unsigned 32-bit big-endian values.
        """
        # Store the count first, then one fixed-width unsigned value per item.
        # Prefix the item count to distinguish sequences whose byte payloads
        # might otherwise be ambiguous when embedded in larger HMAC messages.
        packed = bytearray(struct.pack(">I", len(values)))
        # Fixed-width fields preserve token boundaries.  The mask defines the
        # serializer modulo 2**32, matching an unsigned uint32 representation.
        for value in values:
            packed.extend(struct.pack(">I", int(value) & 0xFFFFFFFF))
        return bytes(packed)

    @staticmethod
    def _digest_to_unit(digest: bytes) -> float:
        """Map an HMAC digest to a float in ``(0, 1)`` via the 64-bit midpoint.

        Used by ``uniform`` and ``uniform_tokens``. HMAC-SHA256 returns 32
        bytes; DISC needs ``y`` in ``(0, 1)``. First 8 bytes become integer
        ``k``, then ``(k + 0.5) / 2**64``.

        Args:
            digest: bytes: A 32-byte HMAC-SHA256 digest; only its first 8 bytes are used.

        Returns:
            float: One scalar (k + 0.5) / 2**64, where k is the first 8 bytes interpreted
            big-endian; mathematically in (0, 1), but rounding can yield 1.0.
        """
        # Taking a fixed eight-byte prefix makes this conversion portable and
        # gives exactly 2**64 conceptual buckets before floating-point rounding.
        integer = int.from_bytes(digest[:8], "big")
        # Place the sample at the bucket midpoint.  Adding 0.5 avoids exact
        # mathematical endpoints, which would be awkward for logarithms and
        # half-open interval membership tests.
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
            tokens: Sequence[int]: Flat sequence of N unsigned 32-bit token IDs, possibly
                empty; N < 2**32.
            extra_bits: Sequence[int] | None: Optional flat 0/1 sequence of length B <
                2**32. None omits this field; an empty sequence encodes a zero-length field.
            domain: bytes | None: Byte-string domain separator of arbitrary length; None
                uses this method’s default domain.

        Returns:
            float: One deterministic midpoint-map sample, mathematically in (0, 1); float
            rounding can yield 1.0.
        """
        # CABS can override the normal token domain while reusing this encoder.
        # Token hashing has its own domain so the same serialized bytes cannot
        # collide with the bit-oriented PRF use under the same secret key.
        tag = domain if domain is not None else self.domain_tokens
        message = tag + self._pack_u32s(tokens)
        # ``None`` means omit the field entirely, whereas [] deliberately adds
        # an encoded zero-length bit sequence; callers can distinguish them.
        if extra_bits is not None:
            message += self._pack_bits(extra_bits)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()
        return self._digest_to_unit(digest)

    def integer_hash(self, tokens: Sequence[int], *, domain: bytes | None = None) -> int:
        """Non-negative integer hash of a token list (CABS ``Hash(Q)``).

        Args:
            tokens: Sequence[int]: Flat window of N unsigned 32-bit token IDs, possibly
                empty; N < 2**32.
            domain: bytes | None: Byte-string domain separator of arbitrary length; None
                uses this method’s default domain.

        Returns:
            int: One unsigned 64-bit scalar in [0, 2**64), from the first eight digest bytes
            in big-endian order.
        """
        # The hash is used only for deterministic CABS frame-boundary tests.
        # Frame hashing is separated from DISC sampling and CABS position
        # selection even though every operation may share one secret key.
        tag = domain if domain is not None else self.domain_cabs_frame
        digest = hmac.new(self._key, tag + self._pack_u32s(tokens), hashlib.sha256).digest()
        # CABS needs an integer for modular boundary tests, so retain the raw
        # unsigned 64-bit value instead of mapping it to the unit interval.
        return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Shared interval / binary-LM math (encode_bit and detect both use these)
# ---------------------------------------------------------------------------


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
        probabilities: Sequence[float] | np.ndarray: One-dimensional vector of shape
            (V,), V >= 2, indexed by token ID. Entries must be finite and nonnegative
            with positive total mass; normalization is performed internally.
        prefix: int: Scalar formed by the bit_index already chosen MSB-first bits, in
            [0, 2**bit_index); 0 at the root.
        bit_index: int: Scalar offset in [0, bit_length), where 0 is the MSB.
        bit_length: int | None: Positive scalar token width; None infers ceil(log2(V)).
            Use a width large enough to represent all V token IDs.

    Returns:
        float: One conditional probability in [0, 1]; returns 0.0 when the selected
        prefix has zero mass.
    """

    # Convert lists and arrays to one predictable numeric representation.  The
    # resulting shape is (V,), where array index equals the vocabulary token ID.
    probs = np.asarray(probabilities, dtype=np.float64)  # shape (V,), V = vocab size
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError("probabilities must be a one-dimensional vocabulary distribution")
    if np.any(probs < 0) or not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("probabilities must be finite, non-negative, and have positive mass")
    # Normalize once so the two child branches are compared as conditional
    # masses under the current prefix.
    probs = probs / probs.sum()
    # A complete binary tree may contain unused leaves when V is not a power
    # of two; those leaves implicitly have zero probability mass.
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
    # Slices naturally omit nonexistent vocabulary leaves in non-power-of-two
    # vocabularies, treating their probability as zero.
    zero_mass = probs[zero_start : min(zero_start + span, probs.size)].sum()
    one_mass = probs[one_start : min(one_start + span, probs.size)].sum()
    # Renormalize inside the selected prefix node.  A zero-mass node should
    # normally be unreachable; returning zero keeps the helper total and lets
    # the caller avoid selecting another nonexistent leaf.
    total = zero_mass + one_mass
    return float(one_mass / total) if total > 0 else 0.0


def token_ids_to_bits(token_ids: Sequence[int], bit_length: int) -> list[int]:
    """Expand token IDs into a flat MSB-first bit string.

    Args:
        token_ids: Sequence[int]: Flat sequence of N_tokens IDs, each in [0,
            2**bit_length).
        bit_length: int: Positive scalar w, the fixed number of MSB-first bits per
            token. For aligned streams, N_bits = N_tokens * w.

    Returns:
        list[int]: Flat 0/1 list of length N_tokens * bit_length, preserving token order
        and writing each token MSB-first with leading zeros.
    """
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    # Keep each token at a stable width so concatenated bits can be decoded.
    result: list[int] = []
    # Validate before extending so every output chunk has exactly bit_length
    # elements and can be inverted without delimiters.
    for token_id in token_ids:
        if not 0 <= token_id < 2**bit_length:
            raise ValueError(f"token id {token_id} does not fit in {bit_length} bits")
        # Zero-padding and left-to-right iteration produce an MSB-first chunk.
        result.extend(int(bit) for bit in f"{token_id:0{bit_length}b}")
    return result


def bits_to_token_ids(bits: Sequence[int], bit_length: int) -> list[int]:
    """Inverse of ``token_ids_to_bits``: group bits into token IDs.

    Args:
        bits: Sequence[int]: Flat MSB-first 0/1 sequence of length N_bits divisible by
            bit_length.
        bit_length: int: Positive scalar w, the fixed number of MSB-first bits per
            token. For aligned streams, N_bits = N_tokens * w.

    Returns:
        list[int]: Flat list of N_bits // bit_length IDs, each in [0, 2**bit_length), in
        original token order.
    """
    # Validate values before parsing text so malformed binary input produces a
    # clear domain error instead of an indirect integer-conversion exception.
    _validate_bits(bits)
    if bit_length <= 0 or len(bits) % bit_length:
        raise ValueError("bit sequence length must be divisible by bit_length")
    # Group the flat stream into fixed-width MSB-first chunks and parse each
    # chunk as one integer token ID.
    # Every fixed-width slice represents one token, with the first sequence
    # bit interpreted as that token's most-significant bit.
    return [
        int("".join(map(str, bits[i : i + bit_length])), 2) for i in range(0, len(bits), bit_length)
    ]


def in_shifted_interval(y: float, probability_one: float, delta_m: float) -> bool:
    """Membership in A_2(M), Equations (21)--(22), on the unit circle.

    The interval of length ``probability_one`` starts at ``delta_m`` and wraps
    around 0 if needed. Bit 1 is emitted iff ``y`` lands in that interval.

    Args:
        y: float: Scalar PRF sample in [0, 1).
        probability_one: float: Scalar P(next bit = 1) in [0, 1].
        delta_m: float: Scalar interval shift; its fractional part gives the position on
            the unit circle.

    Returns:
        bool: One scalar indicating whether y lies in the half-open circular interval of
        length probability_one starting at delta_m % 1.
    """

    # Validate the unit-interval quantities before modular arithmetic; the
    # message shift itself may be any real value because it is reduced below.
    if not 0.0 <= y < 1.0 or not 0.0 <= probability_one <= 1.0:
        raise ValueError("y and probability_one must lie in [0, 1]")
    # Normalize the shift to the unit circle; the interval may wrap at 1.
    start = delta_m % 1.0  # e.g. 0.9
    end = start + probability_one  # e.g. 1.2, which wraps
    # A non-wrapping interval is one ordinary comparison.  A wrapping interval
    # is the union of its tail near 1 and its head near 0.
    return start <= y < end if end <= 1.0 else y >= start or y < end - 1.0


def disc_score(bit: int, y: float, delta_m: float) -> float:
    """DISC score from Equation (24).

    For bit 1 the circular distance is ``(y - delta_m) mod 1``; for bit 0 it
    is ``(delta_m - y) mod 1``. The score is ``-log(distance)``. Watermarked
    bits tend to land near ``delta_m`` on the matching side, so distances are
    small and scores are large.

    Args:
        bit: int: One observed binary scalar, 0 or 1.
        y: float: Scalar PRF sample in [0, 1).
        delta_m: float: Scalar interval shift; its fractional part gives the position on
            the unit circle.

    Returns:
        float: One nonnegative score in nats, -log of the circular distance clamped
        below at float64.tiny.
    """

    # Reject invalid observations before choosing the bit-dependent direction
    # around the circle.
    if bit not in (0, 1) or not 0.0 <= y < 1.0:
        raise ValueError("bit must be binary and y must lie in [0, 1)")
    # Measure distance in the direction associated with the observed bit.
    distance = ((y - delta_m) % 1.0) if bit else ((delta_m - y) % 1.0)
    # PRF midpoint output makes zero probability negligible; guard exact test inputs.
    return -math.log(max(distance, np.finfo(np.float64).tiny))


# ---------------------------------------------------------------------------
# Encoder (Algorithm 3)
# ---------------------------------------------------------------------------


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
        use_prefix: If False, skip ``R``: ``n_star = 0`` and ``y = F([], S)``.
            ``None`` (default) is True for prefix_* modes and False otherwise.
        context_width: ``h`` in real tokens (papers). Binary context is
            ``h * bits_per_token`` bits, with ``bits_per_token = ceil(log2 |V|)``.
            Default ``4``. For encode_bit-only streams, ``bits_per_token=1``.
        bits_per_token: ``ceil(log2 |V|)``. Default ``1`` (each bit is a token).
            ``encode_token`` sets this from the vocabulary size.
        use_cabs: Enable CABS. Default ``True`` iff ``n_positions > 1``.
        cabs_config: Optional ``CabsConfig``. Paper defaults if omitted.
        randomization_window: Number of eligible no-prefix bits per window.
            Defaults to ``context_width``. In ``encode_bit`` mode each bit has
            probability ``1 / randomization_window`` of being sampled from the
            LM; ``encode_token`` makes one selector decision per real token.
            ``0`` disables this randomization.
        randomize_without_prefix: If ``True`` and ``use_prefix=False``, enable
            the optional window-based LM randomization. ``False`` preserves the
            deterministic no-prefix behavior.
        entropy_threshold: Prefix-stop in nats. Ignored when ``use_prefix``
            is False. Default ``5.0``.
        seed: RNG seed for unwatermarked / random-init bits.
        message_mapping: ``"direct"`` or ``"gray"`` (applied per symbol).

    Responsibilities:
        The encoder owns the evolving stream state: ``bits``, ``tokens``, the
        random-prefix length, empirical entropy, and optional CABS state. It
        does not store the payload as literal payload bits in the output. It
        converts the payload into interval shifts and uses those shifts while
        sampling each LM bit.

    State transitions for one bit::

        input probability_one
          -> decide CABS position, if enabled
          -> decide prefix initialization and context warm-up
          -> ordinary Bernoulli sample based on the LM distribution when initialization is required
          -> otherwise compute y = prf_uniform(...)
          -> bit = in_shifted_interval(y, probability_one, delta_m)
          -> append bit and update token/CABS state

    ``encode_token`` is the vocabulary-level entry point. It walks the binary
    representation of one token and calls ``encode_bit`` once per token bit.
    ``encode_bit`` is the lower-level entry point for already-binary streams.
    The two entry points share the same PRF and interval logic.
    """

    def __init__(
        self,
        key: bytes | str | int,
        payload: int,
        payload_bits: int,
        *,
        n_positions: int = 1,
        context_mode: ContextMode = "prefix_bit_ngram",
        use_prefix: bool | None = None,
        entropy_threshold: float = 5.0,
        context_width: int = 4,
        bits_per_token: int = 1,
        seed: int | None = None,
        message_mapping: MessageMapping = "direct",
        use_cabs: bool | None = None,
        cabs_config: CabsConfig | None = None,
        randomization_window: int | None = None,
        randomize_without_prefix: bool = False,
    ):
        """Initialize encoder configuration and empty stream state.

        This constructor validates the payload and algorithm settings, creates
        the shared ``HmacPrf``, splits the payload into per-position symbols,
        and derives each position's interval shift. Encoding itself happens
        later through ``encode_bit`` or ``encode_token``.

        Raises:
            ValueError: If dimensions, payload, context mode, or configuration
                values are invalid.

        Args:
            key: bytes | str | int: Nonempty byte string or UTF-8 text of arbitrary length,
                or a nonnegative scalar integer encoded as at least one big-endian byte.
            payload: int: Scalar full message in [0, 2**(payload_bits * n_positions)).
            payload_bits: int: Positive scalar m, the number of message bits per DISC
                position; each symbol lies in [0, 2**m).
            n_positions: int: Positive scalar H, the number of DISC positions.
            context_mode: ContextMode: One string: "prefix_bit_ngram", "bit_ngram",
                "token_ngram", or "prefix_token_ngram"; selects the PRF context
                representation.
            use_prefix: bool | None: Scalar prefix flag. None follows context_mode; False
                uses an empty prefix. True requires a prefix_* mode.
            entropy_threshold: float: Nonnegative scalar prefix-stop threshold in nats;
                ignored without a prefix.
            context_width: int: Positive scalar h, the context length in real tokens.
            bits_per_token: int: Positive scalar w, initially 1 for raw bit streams;
                encode_token sets it from vocabulary size.
            seed: int | None: Scalar seed for ordinary Bernoulli sampling; None uses
                random.Random default seeding.
            message_mapping: MessageMapping: One string, "direct" or "gray", applied
                separately to each position symbol.
            use_cabs: bool | None: Scalar scheduler flag; None enables CABS exactly when
                n_positions > 1.
            cabs_config: CabsConfig | None: One scheduler configuration object; None selects
                CABS defaults.
            randomization_window: int | None: Nonnegative scalar selector window; None uses
                context_width, 0 disables selection, and positive W selects ordinary
                sampling with probability 1/W.
            randomize_without_prefix: bool: Scalar flag enabling the separate selector PRF
                when use_prefix is False.

        Returns:
            None: Initializes this encoder with empty bit/token/mask lists and n_positions
            symbols and interval shifts.
        """
        if n_positions < 1 or payload_bits < 1:
            raise ValueError("n_positions and payload_bits must be positive")
        total_bits = payload_bits * n_positions
        if not 0 <= payload < 2**total_bits:
            raise ValueError("payload must be in [0, 2**(payload_bits * n_positions))")
        if entropy_threshold < 0 or context_width < 1 or bits_per_token < 1:
            raise ValueError(
                "entropy_threshold must be non-negative and context_width, bits_per_token positive"
            )
        if context_mode not in (
            "prefix_bit_ngram",
            "bit_ngram",
            "token_ngram",
            "prefix_token_ngram",
        ):
            raise ValueError("unknown context_mode")
        self.prf = HmacPrf(key)
        # Separate PRF instance for deciding whether an R=[] token/bit is
        # watermarked. Its domain is distinct from the DISC y-generation PRF.
        self.selector_prf = HmacPrf(key)
        self.payload = payload
        self.payload_bits = payload_bits
        self.n_positions = n_positions
        self.context_mode: ContextMode = context_mode
        self.use_prefix = _resolve_use_prefix(context_mode, use_prefix)
        # Store both the original symbols and the mapped symbols so detection
        # can expose the user's payload while scoring the encoded intervals.
        self.message_mapping: MessageMapping = message_mapping
        self.symbols = split_payload(payload, n_positions, payload_bits)
        self.mapped_symbols = [_map_payload(symbol, message_mapping) for symbol in self.symbols]
        self.position_deltas = [mapped / 2**payload_bits for mapped in self.mapped_symbols]
        # Backward-compatible aliases for the single-position (H=1) case.
        self.mapped_payload = self.mapped_symbols[0]
        self.delta_m = self.position_deltas[0]
        self.entropy_threshold = entropy_threshold
        self.context_width = context_width  # h, real tokens
        self.bits_per_token = bits_per_token  # w = ceil(log2 |V|); 1 for bit streams
        self.rng = random.Random(seed)
        self.bits: list[int] = []
        self.tokens: list[int] = []
        self.token_positions: list[int | None] = []
        self.empirical_entropy = 0.0
        # ``None`` means the prefix boundary has not been reached; no-prefix
        # modes use zero because their PRF input is always R = [].
        self.n_star: int | None = None if self.use_prefix else 0
        self.n_star_tokens: int | None = (
            None if self.use_prefix and context_mode == "prefix_token_ngram" else 0
        )
        self._in_token = False
        self._token_width = 1
        # Raw encode_bit calls can still represent real tokens when
        # bits_per_token > 1.  This holds the one CABS position selected at the
        # start of that token until all of its bits have been emitted.
        self._active_cabs_position: int | None = None
        # ``None`` is also a valid CABS result for an ineligible token, so keep
        # a separate flag indicating whether this real token was already
        # proposed. This prevents retrying propose once per bit.
        self._active_cabs_decided = False
        self._active_delta = self.delta_m
        self.use_cabs = n_positions > 1 if use_cabs is None else use_cabs
        # Retain the caller's real-token configuration so encode_token can
        # rebuild binary CABS after it learns the vocabulary width w.
        self.cabs_config = cabs_config
        if randomization_window is None:
            randomization_window = context_width
        if randomization_window < 0:
            raise ValueError("randomization_window must be non-negative")
        self.randomization_window = randomization_window
        self.randomize_without_prefix = randomize_without_prefix
        self.watermark_mask: list[bool] = []
        self._seen_watermark_contexts: set[tuple[object, ...]] = set()
        self.cabs: CabsScheduler | None = None
        if self.use_cabs:
            if context_mode in ("prefix_bit_ngram", "bit_ngram"):
                # Binary-context DISC schedules individual bits, so CABS uses
                # h*w-bit eligibility contexts and bit-scaled frame settings.
                cabs_h, effective_config = _binary_cabs_settings(
                    cabs_config, n_positions, context_width, bits_per_token
                )
                self.cabs = CabsScheduler(self.prf, n_positions, cabs_h, effective_config)
            else:
                self.cabs = CabsScheduler(self.prf, n_positions, context_width, cabs_config)

    @property
    def random_initialization(self) -> list[int]:
        """Return the unwatermarked prefix ``R`` as ``list[int]`` of 0/1.

        This is ``bits[:n_star]`` once the encoder has finished the LM-sampled
        prefix. If that prefix is still being generated, this is all bits so far.

        This is a ``@property`` so callers read it as state
        (``encoder.random_initialization``), while the returned value remains
        derived from the current ``bits`` and ``n_star``.

        Args:
            None. Reads the current instance state.

        Returns:
            list[int]: New flat 0/1 list of length n_star once fixed, otherwise
            len(self.bits); empty when prefixes are disabled.
        """
        # While initialization is open, every bit generated so far belongs to
        # R.  Once the boundary is fixed, later watermarked bits must be kept
        # out of the returned prefix.
        end = len(self.bits) if self.n_star is None else self.n_star
        # Return a slice rather than the internal list so callers cannot mutate
        # the encoder's accumulated history through this property.
        return self.bits[:end]

    @property
    def binary_context_width(self) -> int:
        """Binary n-gram length: ``h * ceil(log2 |V|)`` bits.

        Args:
            None. Reads the current instance state.

        Returns:
            int: One positive scalar equal to context_width * bits_per_token, measured in
            bits.
        """
        # ``context_width`` is expressed in real tokens, so bit modes expand
        # it by the binary width of one token.
        return self.context_width * self.bits_per_token

    def _prf_context_len(self) -> int:
        """Units expected by ``prf_uniform`` for the current context_mode.

        Args:
            None. Reads the current instance state.

        Returns:
            int: One scalar context length: context_width token IDs in token modes,
            otherwise binary_context_width bits.
        """
        # Token modes express h directly in token IDs.  Bit modes must expand
        # each of those h tokens into its fixed-width binary representation.
        if self.context_mode in ("token_ngram", "prefix_token_ngram"):
            return self.context_width
        return self.binary_context_width

    def _cabs_eligibility_context(self) -> Sequence[int] | None:
        """Return the complete DISC PRF context for the next CABS decision.

        CABS assigns one real token to a position, but eligibility must prevent
        reuse of the PRF input that produces DISC's ``y``. Bit modes therefore
        use the preceding ``h * w`` bits; token modes use the preceding ``h``
        token IDs. ``None`` denotes an incomplete warm-up context.

        Args:
            None. Reads the current encoder bit and token histories.

        Returns:
            Sequence[int] | None: A flat context of length ``h * w`` binary
            values for bit modes or length ``h`` token IDs for token modes;
            ``None`` until the required history is available.
        """
        if self.context_mode in ("prefix_bit_ngram", "bit_ngram"):
            width = self.binary_context_width
            return self.bits[-width:] if len(self.bits) >= width else None
        return self.tokens[-self.context_width :] if len(self.tokens) >= self.context_width else None

    def _needs_random_init(self) -> bool:
        """Return whether the encoder must continue building prefix ``R``.

        This helper is encoder-only. Prefix modes use an initial unwatermarked
        segment to seed the PRF. The encoder stops initialization when its
        locally accumulated surprisal reaches the configured entropy threshold
        and the required context window exists. Bit-prefix mode checks
        ``self.bits``; token-prefix mode checks ``self.tokens``. The detector
        never calls this helper, has no LLM probabilities, and instead searches
        candidate ``n_star`` or ``n_star_tokens`` values.

        Non-prefix modes return ``False`` because they define ``R = []``.

        Args:
            None. Reads the current instance state.

        Returns:
            bool: One scalar; True means the next sample must extend the ordinary-sampled
            prefix.
        """
        # Empty-prefix configurations never accumulate R, regardless of the
        # entropy or amount of context observed so far.
        if not self.use_prefix:
            return False
        # Bit-prefix initialization remains active until both stopping
        # requirements hold: sufficient surprisal and a complete bit context.
        if self.context_mode == "prefix_bit_ngram":
            return self.n_star is None and (
                self.empirical_entropy < self.entropy_threshold
                or len(self.bits) < self.binary_context_width
            )
        # Token-prefix initialization applies the analogous test at real-token
        # boundaries, while entropy is still accumulated from sampled bits.
        if self.context_mode == "prefix_token_ngram":
            return self.n_star_tokens is None and (
                self.empirical_entropy < self.entropy_threshold or len(self.tokens) < self.context_width
            )
        # This fallback is defensive; prefix use is validated against the two
        # prefix modes when configuration is created.
        return False

    def _needs_context_warmup(self) -> bool:
        """True until the n-gram window exists. Applies with or without ``R``.

        Args:
            None. Reads the current instance state.

        Returns:
            bool: One scalar; True until enough preceding tokens or bits exist for the
            configured context window.
        """
        # The required unit depends on whether the PRF hashes token IDs or
        # their binary expansion.
        # Token-history length and binary-history length are intentionally
        # checked in different units; mixing them would shift the first
        # watermarked sample by a factor of bits_per_token.
        if self.context_mode in ("token_ngram", "prefix_token_ngram"):
            return len(self.tokens) < self.context_width
        return len(self.bits) < self.binary_context_width

    def _sample_unwatermarked(self, probability_one: float) -> int:
        """Sample one ordinary Bernoulli bit and update prefix statistics.

        The sample is drawn from the supplied language-model probability,
        appended to ``self.bits``, and contributes its surprisal to
        ``self.empirical_entropy``. This is used for prefix initialization,
        context warm-up, and explicitly unwatermarked tokens.

        Args:
            probability_one: float: Scalar P(next bit = 1) in [0, 1].

        Returns:
            int: One sampled bit, 0 or 1. Appends one bit to self.bits and updates empirical_entropy.
        """
        # Inverse-transform Bernoulli sampling preserves the language model's
        # requested marginal probability without consulting either DISC PRF.
        bit = int(self.rng.random() < probability_one)
        # Surprisal is measured for the value that was actually sampled.  The
        # tiny floor keeps deterministic p=0 or p=1 inputs numerically safe.
        chosen_probability = probability_one if bit else 1.0 - probability_one
        self.empirical_entropy += -math.log(max(chosen_probability, np.finfo(float).tiny))
        # This low-level helper owns the binary history append.  Callers decide
        # separately whether and when token-level state is committed.
        self.bits.append(bit)
        return bit

    def _watermark_context_key(self, bit_in_token: int) -> tuple[object, ...]:
        """Return the hashable PRF context key for the next bit.

        The key mirrors the parts of ``prf_uniform`` that determine the PRF
        input, but it does not compute HMAC. It is used to detect a context
        that has already been watermarked so the next occurrence can be
        sampled from the LM instead of repeating the same PRF result.

        Args:
            bit_in_token: int: Nonnegative scalar bit offset within the current token, with
                0 denoting the MSB.

        Returns:
            tuple[object, ...]: A 3-tuple (mode, prefix_bits, context_bits) for bit modes,
            or 4-tuple (mode, prefix_tokens, history_tokens, bit_in_token) for token modes.
            Nested sequences are tuples; prefix length is n_star or n_star_tokens (zero
            without R), and context length is at most binary_context_width bits or
            context_width tokens.
        """
        # Mirror the exact prefix/context selection performed by prf_uniform;
        # a tuple provides an immutable key suitable for a set membership test.
        if self.context_mode in ("prefix_bit_ngram", "bit_ngram"):
            context = tuple(self.bits[-self.binary_context_width :])
            prefix = tuple(self.bits[: self.n_star or 0]) if self.context_mode == "prefix_bit_ngram" else ()
            return (self.context_mode, prefix, context)
        # Token modes identify the real-token history and also include the bit
        # offset because multiple bits share the same surrounding token n-gram.
        history = tuple(self.tokens[-self.context_width :])
        prefix = (
            tuple(self.tokens[: self.n_star_tokens or 0])
            if self.context_mode == "prefix_token_ngram"
            else ()
        )
        return (self.context_mode, prefix, history, bit_in_token)

    def _scheduled_random_bit(self) -> bool:
        """Return whether the next no-prefix eligible bit is randomized.

        A domain-separated PRF hashes the current context and selects the
        unwatermarked path with probability ``1 / randomization_window``.
        Therefore this decision is reproducible for the same key and context,
        but it is independent of the PRF sample used to generate ``y``.

        Args:
            None. Reads the current instance state.

        Returns:
            bool: One scalar; True selects ordinary sampling, False keeps the watermark
            path. Reads current encoder state without appending samples.
        """
        # The selector is meaningful only in explicitly randomized empty-R
        # operation.  A nonpositive window is the documented off switch.
        if (
            self.use_prefix
            or not self.randomize_without_prefix
            or self.randomization_window <= 0
        ):
            return False
        # Recreate the same context representation the detector will later use
        # when it reconstructs the watermark mask from the observed sequence.
        if self.context_mode in ("prefix_bit_ngram", "bit_ngram"):
            context = self.bits[-self.binary_context_width :]
            selector = self.selector_prf.uniform(
                [], context, domain=HmacPrf.domain_randomization
            )
        else:
            history = self.tokens[-self.context_width :]
            selector = self.selector_prf.uniform_tokens(
                history,
                domain=HmacPrf.domain_randomization,
            )
        # The upper interval has measure 1/W for a uniform selector, so True
        # occurs with exactly the configured ordinary-sampling probability.
        return selector >= 1.0 - (1.0 / self.randomization_window)

    def _round_robin_position(self) -> int | None:
        """Return the no-CABS position for the token containing the next bit.

        CABS-off multi-position DISC assigns complete real tokens in a stable
        round-robin order. Prefix-token mode starts that order after its
        complete-token prefix R; bit-prefix mode intentionally counts from
        token zero because R may end inside a token and the detector groups
        that entire token by its original boundary.

        Returns:
            ``int`` in ``[0, n_positions)`` when a multi-position token can
            be assigned, else ``None`` while token-prefix R is still open or
            for the single-position configuration.
        """
        if self.n_positions == 1 or self.use_cabs:
            return None
        # During encode_token the current token is not in self.tokens yet.
        # Raw encode_bit reconstructs its current real-token index from w.
        token_index = (
            len(self.tokens)
            if self._in_token
            else len(self.bits) // max(self.bits_per_token, 1)
        )
        if self.context_mode == "prefix_token_ngram":
            # Until an entire token-prefix R has ended, this token is an
            # ordinary sample and has no scheduling origin yet.
            if self.n_star_tokens is None:
                return None
            token_index -= self.n_star_tokens
        return token_index % self.n_positions

    def encode_bit(self, probability_one: float, *, watermark: bool = True, delta_m: float | None = None) -> int:
        """Sample the next binary token given P(bit = 1).

        During the unwatermarked prefix, or when ``watermark=False``, samples
        Bernoulli. Once ``R`` is finished, draws ``y`` via ``prf_uniform`` and
        uses the shifted-interval encoder (Algorithm 3) with ``delta_m``
        (default: the active position's shift). See the module call flow.

        Call flow::

            encode_bit
              ├─ CabsScheduler.propose (only at token boundaries)
              ├─ _needs_random_init / _needs_context_warmup
              ├─ _sample_unwatermarked, if either warm-up condition is true
              └─ otherwise:
                   prf_uniform -> HmacPrf.uniform or uniform_tokens
                   in_shifted_interval -> append the selected bit

        Side effects:
            Appends to ``self.bits``. At a CABS token boundary it also appends
            to ``self.tokens`` and ``self.token_positions`` and commits the
            completed token to the scheduler.

        Args:
            probability_one: float: Scalar P(next bit = 1) in [0, 1].
            watermark: bool: Scalar flag; False forces ordinary Bernoulli sampling.
            delta_m: float | None: Scalar circular interval shift; None uses the active
                position’s shift.

        Returns:
            int: One bit, 0 or 1; appends one entry to self.bits and self.watermark_mask and
            updates encoder state.
        """
        if not 0.0 <= probability_one <= 1.0:
            raise ValueError("probability_one must lie in [0, 1]")
        cabs_position: int | None = self._active_cabs_position
        # Keep a local non-optional reference for branches where raw CABS is
        # active. This also makes the invariant explicit to static checkers:
        # propose/commit are never called when the scheduler is absent.
        cabs_scheduler = self.cabs
        # With H>1 and CABS disabled, encode and detect both assign entire
        # real tokens deterministically. This is separate from CABS: no
        # eligibility filtering, queue, frame state, or PRF tie-break occurs.
        round_robin_position = self._round_robin_position()
        if round_robin_position is not None and delta_m is None:
            delta_m = self.position_deltas[round_robin_position]
        binary_cabs = (
            cabs_scheduler is not None
            and self.context_mode in ("prefix_bit_ngram", "bit_ngram")
        )
        if binary_cabs:
            # In binary DISC modes every generated bit is a CABS scheduling
            # token. Its eligibility key is the complete h*w-bit PRF context.
            cabs_prefix_active = self.use_prefix and self._needs_random_init()
            cabs_position = None
            if not cabs_prefix_active:
                cabs_context = self._cabs_eligibility_context()
                if cabs_context is not None:
                    assert cabs_scheduler is not None
                    cabs_position = cabs_scheduler.propose(
                        self.bits, eligibility_context=cabs_context
                    )
            if cabs_position is None:
                watermark = False
            elif delta_m is None:
                delta_m = self.position_deltas[cabs_position]
        # A raw encode_bit stream is grouped into real tokens of width w. CABS
        # is consulted only at offset zero, never once per individual bit.
        raw_cabs = not self._in_token and cabs_scheduler is not None and not binary_cabs
        raw_token_offset = len(self.bits) % max(self.bits_per_token, 1)
        if raw_cabs and raw_token_offset == 0:
            # Prefix-token R is generated before CABS begins. Skipping proposal
            # prevents R tokens from consuming positions or frame state.
            cabs_prefix_active = self.use_prefix and self._needs_random_init()
            self._active_cabs_position = None
            self._active_cabs_decided = not cabs_prefix_active
            cabs_position = None
            if not cabs_prefix_active:
                # Select one position for the whole real token. An ineligible
                # token remains ordinary-sampled and receives no position.
                assert cabs_scheduler is not None
                cabs_context = self._cabs_eligibility_context()
                cabs_position = (
                    cabs_scheduler.propose(self.tokens, eligibility_context=cabs_context)
                    if cabs_context is not None
                    else None
                )
                self._active_cabs_position = cabs_position
                if cabs_position is None:
                    watermark = False
                elif delta_m is None:
                    delta_m = self.position_deltas[cabs_position]
            else:
                watermark = False
        elif raw_cabs:
            # Reuse the position chosen for this token's first bit. If that
            # token began inside R, wait until R ends inside this token before
            # proposing it; the eventual position still belongs to the whole
            # real token and is committed only after its final bit.
            cabs_position = self._active_cabs_position
            if (
                cabs_position is None
                and self.context_mode == "prefix_bit_ngram"
                and self.n_star is not None
                and not self._active_cabs_decided
            ):
                assert cabs_scheduler is not None
                cabs_context = self._cabs_eligibility_context()
                cabs_position = (
                    cabs_scheduler.propose(self.tokens, eligibility_context=cabs_context)
                    if cabs_context is not None
                    else None
                )
                self._active_cabs_position = cabs_position
                self._active_cabs_decided = True
            if cabs_position is None:
                watermark = False
            elif delta_m is None:
                delta_m = self.position_deltas[cabs_position]

        # Prefix initialization and n-gram warm-up both require ordinary LM
        # samples. ``watermark=False`` also forces the ordinary path.
        unwatermarked = self._needs_random_init() or self._needs_context_warmup() or not watermark
        bit_in_token = len(self.bits) % max(self._token_width, 1) if self._in_token else 0
        context_key: tuple[object, ...] | None = None
        if not unwatermarked and not self._in_token:
            # In raw-bit mode, the optional selector PRF and repeated-context
            # policy decide whether this bit may use the DISC watermark.
            context_key = self._watermark_context_key(bit_in_token)
            unwatermarked = (
                self._scheduled_random_bit()
                or context_key in self._seen_watermark_contexts
            )
        if not unwatermarked and self.use_prefix and self.context_mode == "prefix_bit_ngram" and self.n_star is None:
            # The first eligible bit after warm-up marks the end of R.
            self.n_star = len(self.bits)

        if unwatermarked:
            # Extend R or the context window with an ordinary Bernoulli draw.
            bit = self._sample_unwatermarked(probability_one)
            if (
                self.context_mode == "prefix_bit_ngram"
                and self.n_star is None
                and self.empirical_entropy >= self.entropy_threshold
                and len(self.bits) >= self.binary_context_width
            ):
                # Record the completed prefix after this sample is included.
                self.n_star = len(self.bits)
        else:
            # Derive y from the configured prefix/context and shift the
            # Bernoulli interval by the payload-dependent DISC value. This is
            # the second PRF when selector mode is enabled, not the selector.
            shift = self.delta_m if delta_m is None else delta_m
            token_index = len(self.tokens)
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=self.bits,
                bit_index=len(self.bits),
                context_width=self._prf_context_len(),
                n_star=0 if self.n_star is None else self.n_star,
                tokens=self.tokens,
                token_index=token_index,
                bit_in_token=bit_in_token,
                n_star_tokens=0 if self.n_star_tokens is None else self.n_star_tokens,
            )
            bit = int(in_shifted_interval(y, probability_one, shift))
            self.bits.append(bit)

            if context_key is not None:
                self._seen_watermark_contexts.add(context_key)

        self.watermark_mask.append(not unwatermarked)

        if binary_cabs:
            # CABS frame bookkeeping uses the binary token itself in this mode.
            assert cabs_scheduler is not None
            cabs_scheduler.commit(bit)
        elif raw_cabs:
            # Commit CABS only after the complete real token has been formed.
            # The reconstructed token ID is used for CABS history even though
            # the DISC context itself may remain binary.
            token_width = max(self.bits_per_token, 1)
            if raw_token_offset == token_width - 1:
                token_bits = self.bits[-token_width:]
                token_id = int("".join(map(str, token_bits)), 2)
                self.tokens.append(token_id)
                self.token_positions.append(self._active_cabs_position)
                assert cabs_scheduler is not None
                cabs_scheduler.commit(token_id)
                self._active_cabs_position = None
                self._active_cabs_decided = False
        elif not self._in_token and self.context_mode in ("token_ngram", "prefix_token_ngram"):
            # Raw encode_bit has no external real-token IDs. For token-context
            # simulation, treat this binary token as one token so the next
            # call has the same token history that detection reconstructs.
            self.tokens.append(bit)
            self.token_positions.append(None)
            if (
                self.use_prefix
                and self.context_mode == "prefix_token_ngram"
                and self.n_star_tokens is None
                and self.empirical_entropy >= self.entropy_threshold
                and len(self.tokens) >= self.context_width
            ):
                # Raw bit simulation treats one bit as one token, so record
                # the token-prefix boundary after this ordinary sample.
                self.n_star_tokens = len(self.tokens)
                self.n_star = len(self.bits)
        return bit

    def encode_token(self, probabilities: Sequence[float] | np.ndarray) -> int:
        """Sample one vocabulary token by walking the binarized LM tree.

        With CABS, the token is assigned to a position first; all of its bits
        use that position's DISC shift. Ineligible or warm-up tokens are
        sampled without a watermark.

        Call flow::

            encode_token
              -> validate the probability vector and determine token width
              -> choose the CABS position for the complete token
              -> for each bit from most significant to least significant:
                   conditional_bit_probability
                   encode_bit
              -> reject a zero-probability leaf
              -> append the completed token and commit it to CABS

        The partial integer ``prefix`` is passed to
        ``conditional_bit_probability`` so each successive bit is sampled
        from the correct conditional distribution. ``encode_bit`` still owns
        the common warm-up, PRF, interval-shift, and bit-history behavior.

        Args:
            probabilities: Sequence[float] | np.ndarray: One-dimensional vector of shape
                (V,), V >= 2, indexed by token ID. Entries must be finite and nonnegative
                with positive total mass; normalization is performed internally.

        Returns:
            int: One token ID in [0, V). Appends ceil(log2(V)) MSB-first bits and mask
            entries, and one token and position entry to encoder state.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        # A vocabulary of V IDs needs ceil(log2(V)) bits in the binary tree.
        width = math.ceil(math.log2(probs.size))
        if self.bits_per_token not in (1, width):
            raise ValueError(
                f"bits_per_token is {self.bits_per_token} but this vocab needs {width} bits"
            )
        self.bits_per_token = width
        self._token_width = width
        if (
            self.cabs is not None
            and self.context_mode in ("prefix_bit_ngram", "bit_ngram")
            and not self.bits
        ):
            # ``encode_token`` learns w from the vocabulary. Rebuild the empty
            # binary scheduler so its h*w context and frame scales use that
            # actual width instead of the constructor's provisional width.
            cabs_h, effective_config = _binary_cabs_settings(
                self.cabs_config, self.n_positions, self.context_width, width
            )
            self.cabs = CabsScheduler(self.prf, self.n_positions, cabs_h, effective_config)
        # Prefix tokens are ordinary LM samples and have no DISC position.  A
        # position is selected only once this token is after the completed R.
        prefix_active = self.use_prefix and self._needs_random_init()
        # CABS-off H>1 uses one position for the entire real token. For a
        # prefix-bit stream this is known even if R ends within the token;
        # early R bits remain ordinary while later bits use this position.
        position = self._round_robin_position()
        if position is None and not prefix_active and self.context_mode not in (
            "prefix_bit_ngram",
            "bit_ngram",
        ):
            # Single-position legacy behavior retains the first symbol.
            position = 0
        watermark = True
        delta = self.delta_m if position is None else self.position_deltas[position]
        if (
            self.cabs is not None
            and self.context_mode not in ("prefix_bit_ngram", "bit_ngram")
            and not prefix_active
        ):
            # The position is selected once for the whole token, then reused
            # for every bit generated below.
            cabs_context = self._cabs_eligibility_context()
            position = (
                self.cabs.propose(self.tokens, eligibility_context=cabs_context)
                if cabs_context is not None
                else None
            )
            watermark = position is not None
            if position is not None:
                delta = self.position_deltas[position]
        elif prefix_active or self._needs_context_warmup():
            watermark = False
        if watermark and not self.use_prefix and self.randomize_without_prefix:
            # Make one selector decision for the whole real token. Each bit
            # below inherits this decision through the watermark argument.
            watermark = not self._scheduled_random_bit()

        self._in_token = True
        prefix = 0
        try:
            for bit_index in range(width):
                # ``prefix`` contains the already selected high bits and is
                # updated after encode_bit returns the next bit.
                p_one = conditional_bit_probability(probs, prefix, bit_index, width)
                prefix = (prefix << 1) | self.encode_bit(
                    p_one, watermark=watermark, delta_m=delta
                )
        finally:
            self._in_token = False
        if prefix >= probs.size:
            raise RuntimeError("binarized sampling reached a zero-probability token")
        # Only append the token after all of its bits have been generated.
        self.tokens.append(prefix)
        self.token_positions.append(position)
        if self.cabs is not None and self.context_mode not in ("prefix_bit_ngram", "bit_ngram"):
            self.cabs.commit(prefix)
        if (
            self.use_prefix
            and self.context_mode == "prefix_token_ngram"
            and self.n_star_tokens is None
            and self.empirical_entropy >= self.entropy_threshold
            and len(self.tokens) >= self.context_width
        ):
            # Token-prefix mode records R in token units, then keeps the bit
            # equivalent for callers that inspect encoder.n_star.
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
        symbols: Decoded symbols as a tuple of H integers in
            ``[0, 2**payload_bits)`` on detection through the position path.
            Example: ``(3, 3)`` for ``H=2``. Otherwise ``None``, including
            successful detection through the single-position scoring path.
        position_p_values: ``tuple[float, ...]`` with one corrected p-value
            in ``[0, 1]`` per nonempty assigned group, in position order;
            length at most H. Empty on the single-position scoring path.
            The position path combines these values with Fisher’s method.
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


# ---------------------------------------------------------------------------
# Detector (Algorithm 4)
# ---------------------------------------------------------------------------


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
        use_prefix: Must match the encoder. False → ``R = []``, no ``n_star``
            search. ``None`` follows the context mode.
        context_width: ``h`` in real tokens. Binary n-gram is ``h * bits_per_token``.
        bits_per_token: ``ceil(log2 |V|)``. Default ``1``. Detection with
            ``token_ids`` uses ``bit_length`` when given.
        fpr: Overall false-positive target. Default ``0.01``.
        use_cabs: Default ``True`` iff ``n_positions > 1``.
        cabs_config: Optional ``CabsConfig``.
        randomization_window: Window size used by the optional second PRF.
            Defaults to ``context_width``.
        randomize_without_prefix: Set to ``True`` when the encoder used the
            second PRF to decide which no-prefix tokens were unwatermarked.

    Responsibilities:
        The detector is the inverse statistical workflow, not a decoder that
        reads a literal payload field. It searches the possible interval
        symbols and, in prefix modes, possible ``n_star`` values. For each
        hypothesis it reconstructs the PRF sample, scores the observed bits,
        applies the Erlang tail and multiple-testing corrections, and returns
        a ``DetectionResult``.

    Single-position call flow::

        detect(bits)
          -> validate bits and determine the context width
          -> enumerate candidate n_star values
          -> for each candidate and mapped payload:
               _score -> prf_uniform -> disc_score
               gammaincc(score, count)
          -> apply payload and start-search corrections
          -> _unmap_payload when a detection passes ``fpr``

    Multi-position call flow::

        detect(bits, token_ids=...)
          -> replay CABS assignments with ``assign_sequence``
          -> group token bit indices by DISC position
          -> _score_indices for every candidate symbol at each position
          -> union-correct each position's best p-value
          -> combine_p_values with Fisher's method
          -> join_payload recovered symbols when detected

    The detector must use the encoder's key, context mode, context width,
    token representation, message mapping, and CABS configuration. A mismatch
    changes the PRF inputs or score grouping and normally prevents recovery.
    """

    def __init__(
        self,
        key: bytes | str | int,
        payload_bits: int,
        *,
        n_positions: int = 1,
        context_mode: ContextMode = "prefix_bit_ngram",
        use_prefix: bool | None = None,
        context_width: int = 4,
        bits_per_token: int = 1,
        fpr: float = 0.01,
        deduplicate_ngrams: bool = True,
        message_mapping: MessageMapping = "direct",
        use_cabs: bool | None = None,
        cabs_config: CabsConfig | None = None,
        randomization_window: int | None = None,
        randomize_without_prefix: bool = False,
    ):
        """Initialize detector configuration for a matching encoder.

        The detector stores the PRF, context, payload-search, statistical-test,
        message-mapping, and optional CABS settings needed by ``detect``. It
        does not retain an observed stream; each call to ``detect`` receives a
        fresh sequence to analyze.

        Raises:
            ValueError: If dimensions, false-positive rate, context mode, or
                other configuration values are invalid.

        Args:
            key: bytes | str | int: Nonempty byte string or UTF-8 text of arbitrary length,
                or a nonnegative scalar integer encoded as at least one big-endian byte.
            payload_bits: int: Positive scalar m, the number of message bits per DISC
                position; each symbol lies in [0, 2**m).
            n_positions: int: Positive scalar H, the number of DISC positions.
            context_mode: ContextMode: One string: "prefix_bit_ngram", "bit_ngram",
                "token_ngram", or "prefix_token_ngram"; selects the PRF context
                representation.
            use_prefix: bool | None: Scalar prefix flag. None follows context_mode; False
                uses an empty prefix. True requires a prefix_* mode.
            context_width: int: Positive scalar h, the context length in real tokens.
            bits_per_token: int: Positive scalar binary width of one token; defaults to 1
                for bit streams.
            fpr: float: Scalar detection threshold strictly between 0 and 1.
            deduplicate_ngrams: bool: Scalar flag that skips repeated context/current-bit
                pairs during scoring.
            message_mapping: MessageMapping: One string, "direct" or "gray", applied
                separately to each position symbol.
            use_cabs: bool | None: Scalar scheduler flag; None enables CABS exactly when
                n_positions > 1.
            cabs_config: CabsConfig | None: One scheduler configuration object; None selects
                CABS defaults.
            randomization_window: int | None: Nonnegative scalar selector window; None uses
                context_width, 0 disables selection, and positive W selects ordinary
                sampling with probability 1/W.
            randomize_without_prefix: bool: Scalar flag enabling the separate selector PRF
                when use_prefix is False.

        Returns:
            None: Initializes this detector’s configuration and keyed PRFs; no observed
            stream is retained.
        """
        # Validate all structural dimensions together before deriving message
        # counts or context lengths from them.
        if payload_bits < 1 or context_width < 1 or n_positions < 1 or bits_per_token < 1:
            raise ValueError(
                "payload_bits, context_width, n_positions, and bits_per_token must be positive"
            )
        # Detection compares a p-value against this threshold, so both endpoint
        # values are excluded: zero can never accept and one always accepts.
        if not 0 < fpr < 1:
            raise ValueError("fpr must lie strictly between 0 and 1")
        # Runtime validation is still needed even though ContextMode helps
        # static type checkers, because Python callers may pass arbitrary text.
        if context_mode not in (
            "prefix_bit_ngram",
            "bit_ngram",
            "token_ngram",
            "prefix_token_ngram",
        ):
            raise ValueError("unknown context_mode")
        # The scoring PRF reconstructs the encoder's y values for every tested
        # prefix and message-symbol hypothesis.
        self.prf = HmacPrf(key)
        # Same key, separate PRF instance and domain for watermark selection.
        self.selector_prf = HmacPrf(key)
        self.payload_bits = payload_bits
        self.n_positions = n_positions
        self.context_mode: ContextMode = context_mode
        self.use_prefix = _resolve_use_prefix(context_mode, use_prefix)
        # Each position contains m bits and therefore has exactly 2**m mapped
        # interval candidates in the exhaustive symbol search.
        self.message_count = 2**payload_bits
        self.context_width = context_width
        self.bits_per_token = bits_per_token
        self.fpr = fpr
        self.deduplicate_ngrams = deduplicate_ngrams
        # Resolve None once so mask reconstruction later works with a concrete
        # nonnegative integer and mirrors the encoder default.
        self.randomization_window = (
            context_width if randomization_window is None else randomization_window
        )
        if self.randomization_window < 0:
            raise ValueError("randomization_window must be non-negative")
        self.randomize_without_prefix = randomize_without_prefix
        # Exercise the mapping helper with a harmless symbol to validate the
        # configuration string before any expensive detection search begins.
        _map_payload(0, message_mapping)
        self.message_mapping: MessageMapping = message_mapping
        # Multi-position encoding normally requires CABS, while an explicit
        # override supports deterministic round-robin assignment for testing.
        self.use_cabs = n_positions > 1 if use_cabs is None else use_cabs
        self.cabs_config = cabs_config
        self._cabs_h = context_width

    def _binary_h(self, bits_per_token: int | None = None) -> int:
        """Return the binary context width corresponding to ``h`` tokens.

        Bit-context modes represent each real token with
        ``bits_per_token`` binary symbols, so the PRF window contains
        ``context_width * bits_per_token`` bits. When no override is supplied,
        the detector's configured width is used.

        Args:
            bits_per_token: int | None: Positive scalar binary token width; None uses the
                configured width.

        Returns:
            int: One positive scalar context length in bits, context_width times the
            effective token width.
        """
        # Prefer a call-specific width when token IDs reveal it; otherwise use
        # the width stored at detector construction.  Multiplication converts
        # h real tokens into the equivalent number of preceding binary bits.
        effective_width = bits_per_token or self.bits_per_token
        return self.context_width * effective_width

    def _prf_context_len(self, bits_per_token: int | None = None) -> int:
        """Return the context-window length in the units used by ``prf_uniform``.

        Token n-gram modes count context in token IDs, while bit n-gram modes
        count context in binary bits. This adapter keeps the detector's call
        sites independent of that representation difference.

        Args:
            bits_per_token: int | None: Positive scalar binary token width; None uses the
                configured width.

        Returns:
            int: One scalar: context_width token IDs in token modes, otherwise
            _binary_h(bits_per_token) bits.
        """
        # prf_uniform interprets this number in mode-specific units.  Passing a
        # bit-expanded value to token mode would accidentally request h*w IDs.
        if self.context_mode in ("token_ngram", "prefix_token_ngram"):
            return self.context_width
        return self._binary_h(bits_per_token)

    def _randomization_mask(
        self,
        bits: Sequence[int],
        tokens: Sequence[int] | None = None,
        bit_length: int = 1,
    ) -> list[bool]:
        """Reproduce the encoder's second-PRF watermark-selection mask.

        ``True`` marks bits eligible for DISC scoring. In probabilistic
        no-prefix mode, the domain-separated selector PRF marks a whole real
        token or a binary bit as unwatermarked with probability
        ``1 / randomization_window``. Prefix modes and deterministic no-prefix
        mode return an all-true mask. This mask only describes the optional
        selector PRF; prefix ``R`` is excluded separately by the scoring
        methods because the detector must not score unwatermarked prefix data.

        Args:
            bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
                and 1 in stream order.
            tokens: Sequence[int] | None: Flat sequence of N_tokens unsigned 32-bit IDs
                aligned with bits. None treats each bit as one token in token modes.
            bit_length: int: Positive scalar w, the fixed number of MSB-first bits per
                token. For aligned streams, N_bits = N_tokens * w.

        Returns:
            list[bool]: Flat mask of length N_bits. False marks selector-excluded bits; True
            still requires the scorer’s prefix, warm-up, and deduplication checks.
        """
        # Start with every bit eligible. The scorer will apply the separate R
        # boundary and remove any bits selected by the second PRF below.
        mask = [True] * len(bits)
        if self.use_prefix or not self.randomize_without_prefix or self.randomization_window <= 0:
            # Deterministic no-prefix mode has no selector exclusions, and
            # prefix modes handle their unwatermarked R in the scorer.
            return mask
        if self.context_mode in ("token_ngram", "prefix_token_ngram"):
            if tokens is None:
                tokens = bits
                bit_length = 1
            for token_index in range(len(tokens)):
                # Token context is the last h IDs before this token. The
                # selector decision is therefore made once per real token.
                history = tokens[max(0, token_index - self.context_width) : token_index]
                selector = self.selector_prf.uniform_tokens(
                    history,
                    domain=HmacPrf.domain_randomization,
                )
                if selector >= 1.0 - (1.0 / self.randomization_window):
                    start = token_index * bit_length
                    # False means intentionally sampled from the LM and must
                    # be ignored by _score_indices.
                    for index in range(start, min(start + bit_length, len(mask))):
                        mask[index] = False
            return mask
        context_width = self._binary_h(bit_length)
        for bit_index in range(len(bits)):
            if bit_index < context_width:
                # No complete context exists yet, so no selector decision is
                # needed; the scorer already excludes these warm-up bits.
                continue
            context = bits[bit_index - context_width : bit_index]
            selector = self.selector_prf.uniform(
                [], context, domain=HmacPrf.domain_randomization
            )
            if selector >= 1.0 - (1.0 / self.randomization_window):
                mask[bit_index] = False
        return mask

    def _score(
        self,
        bits: Sequence[int],
        n_star: int,
        payload: int,
        watermark_mask: Sequence[bool] | None = None,
    ) -> tuple[float, int]:
        """Sum Equation (24) scores for one hypothesized (n_star, payload).

        This is the single-position scoring path. It starts only after the
        context window exists, skips duplicate ``(context, bit)`` n-grams when
        configured, calls ``prf_uniform`` with the hypothesized prefix, and
        accumulates ``disc_score`` for the candidate interval shift.

        Args:
            bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
                and 1 in stream order.
            n_star: int: Nonnegative scalar prefix length in bits; the prefix is
                bits[:n_star].
            payload: int: Scalar candidate mapped symbol in [0, 2**self.payload_bits).
            watermark_mask: Sequence[bool] | None: Flat mask of length N_bits aligned with
                bits; False excludes that bit from scoring. None supplies no explicit
                exclusions.

        Returns:
            tuple[float, int]: Exactly two scalars (score, count): summed score in nats and
            number of scored bits, 0 <= count <= N_bits. No eligible bits gives (0.0, 0).
        """
        # Convert this mapped candidate symbol into the circular interval shift
        # used for every eligible bit under this hypothesis.
        delta_m = payload / self.message_count
        # ``score`` accumulates Equation (24) in nats; ``count`` is the Erlang
        # shape parameter used later to convert that sum into a p-value.
        score = 0.0
        count = 0
        # Deduplication is local to one complete hypothesis.  A different
        # payload or n_star must start with an empty set because its PRF samples
        # and therefore its evidence are evaluated independently.
        seen: set[tuple[int, ...]] = set()
        bit_h = self._binary_h()
        # Watermarked bits start after R, but the n-gram still needs h*w bits.
        # n_star=0 (no R) therefore scores from bit_h, not from index 0.
        start = max(n_star, bit_h)
        # Traverse absolute stream indices so the preceding context slice and
        # optional mask remain aligned with the original observation.
        for index in range(start, len(bits)):
            if watermark_mask is not None and not watermark_mask[index]:
                continue
            # The n-gram identity includes the observed current bit as well as
            # its preceding h*w-bit context, matching the detector's dependence
            # test rather than merely deduplicating PRF inputs.
            context = bits[index - bit_h : index]
            ngram = tuple(context) + (bits[index],)
            if self.deduplicate_ngrams and ngram in seen:
                continue
            seen.add(ngram)
            # Recompute exactly the keyed sample that the encoder used for this
            # bit under the current prefix-length hypothesis.
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=bits,
                bit_index=index,
                context_width=self._prf_context_len(),
                n_star=n_star if self.context_mode == "prefix_bit_ngram" else 0,
            )
            # Add one independent score contribution and record one additional
            # Erlang observation only after all exclusion checks have passed.
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
        watermark_mask: Sequence[bool] | None = None,
    ) -> tuple[float, int]:
        """DISC score a subset of bit indices with one position's symbol.

        This is the multi-position counterpart to ``_score``. The caller has
        already replayed CABS and supplies only the bit indices assigned to
        one position. ``bits`` is still the complete observed stream, but
        ``bit_indices`` selects this position's bits; it is not a second
        independent bitstream. ``tokens`` is the complete token sequence when
        token context is used, and the selected token indices are derived from
        ``bit_indices``. The method reconstructs the matching context, calls
        ``prf_uniform`` for each eligible bit, and sums ``disc_score`` values
        for one candidate symbol.

        Args:
            bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
                and 1 in stream order.
            bit_indices: Sequence[int]: Flat sequence of K absolute indices in [0, N_bits),
                assigned to one position; may be sparse or empty. For balanced assignment K
                is roughly N_bits / H.
            mapped_symbol: int: Scalar mapped position symbol in [0, 2**self.payload_bits).
            n_star: int: Nonnegative scalar prefix length in bits; the prefix is
                bits[:n_star].
            tokens: Sequence[int]: Complete flat sequence of N_tokens unsigned 32-bit IDs
                aligned with bits; token modes use these IDs for context.
            bit_length: int: Positive scalar w, the fixed number of MSB-first bits per
                token. For aligned streams, N_bits = N_tokens * w.
            n_star_tokens: int: Nonnegative scalar prefix length in tokens; used by
                prefix_token_ngram.
            watermark_mask: Sequence[bool] | None: Flat mask of length N_bits aligned with
                bits; False excludes that bit from scoring. None supplies no explicit
                exclusions.

        Returns:
            tuple[float, int]: Exactly two scalars (score, count): summed score in nats and
            number of eligible indices actually scored, 0 <= count <= len(bit_indices).
            Empty selection returns (0.0, 0).
        """
        # This invocation represents ONE position and ONE candidate symbol.
        # ``_detect_positions`` calls it repeatedly for every position/symbol
        # pair, so returning one scalar score here is deliberate.
        delta_m = mapped_symbol / self.message_count
        # Accumulate this position's evidence independently.  The caller later
        # selects its best symbol and combines position-level p-values.
        score = 0.0
        count = 0
        # Repeated evidence is tracked separately for each position/symbol
        # hypothesis; no state is shared between calls to this method.
        seen: set[tuple[int, ...]] = set()
        # ``bit_h`` is always measured in binary bits.  ``prf_h`` is measured
        # in bits for bit modes and real token IDs for token modes.
        bit_h = self._binary_h(bit_length)
        prf_h = self._prf_context_len(bit_length)
        # bit_indices is already the subset assigned to one DISC position by
        # CABS or round robin.  Keep each value absolute so mask, token, and
        # context calculations remain aligned with the complete stream.
        for bit_index in bit_indices:
            # Selector-randomized or otherwise known-unwatermarked bits cannot
            # provide DISC evidence and must not contribute to the count.
            if watermark_mask is not None and not watermark_mask[bit_index]:
                continue
            # Bit-context PRF inputs require h*w preceding bits.  This guard
            # excludes initial warm-up indices whose context slice is incomplete.
            if bit_index < bit_h and self.context_mode in (
                "prefix_bit_ngram",
                "bit_ngram",
            ):
                continue
            # In a prefix-bit stream, CABS may assign the whole token that
            # contains the boundary.  Bits before n_star are still R and must
            # not contribute evidence even though their token has a position.
            if self.context_mode == "prefix_bit_ngram" and bit_index < n_star:
                continue
            # Fixed-width token encoding makes integer division the inverse map
            # from an absolute bit offset to its token and within-token offset.
            token_index = bit_index // bit_length if bit_length else 0
            bit_in_token = bit_index % bit_length if bit_length else 0
            if (
                self.context_mode == "prefix_token_ngram"
                and token_index < n_star_tokens
            ):
                # The decoder hypothesizes tokens[:n_star_tokens] as R. Those
                # tokens were generated without watermarking and are never
                # evidence for or against the payload.
                continue
            # Construct the dependency key in the same units as the selected
            # context mode.  Token modes share history across a token's bits,
            # while bit modes slide by one binary position at a time.
            if self.context_mode in ("token_ngram", "prefix_token_ngram"):
                if token_index < self.context_width:
                    continue
                ngram = tuple(tokens[token_index - self.context_width : token_index]) + (
                    bits[bit_index],
                )
            else:
                ngram = tuple(bits[bit_index - bit_h : bit_index]) + (bits[bit_index],)
            # Only the first occurrence of a repeated context/current-bit pair
            # is retained when deduplication is enabled, avoiding overstated
            # evidence from deterministic repeated PRF inputs.
            if self.deduplicate_ngrams and ngram in seen:
                continue
            seen.add(ngram)
            # Reconstruct y from the complete observed histories plus the
            # current n_star hypothesis.  Token-prefix mode uses n_star_tokens;
            # bit-prefix mode uses n_star directly.
            y = prf_uniform(
                self.prf,
                self.context_mode,
                bits=bits,
                bit_index=bit_index,
                context_width=prf_h,
                n_star=n_star,
                tokens=tokens,
                token_index=token_index,
                bit_in_token=bit_in_token,
                n_star_tokens=n_star_tokens,
            )
            # One accepted index adds exactly one term to the position score
            # and increments the Erlang shape count by one.
            score += disc_score(bits[bit_index], y, delta_m)
            count += 1
        # Return this single position/symbol aggregate.  It is not the total
        # across all positions; that reduction happens in _detect_positions.
        return score, count

    def _detect_positions(
        self,
        bits: list[int],
        token_ids: Sequence[int],
        bit_length: int,
        n_star: int,
        n_star_tokens: int,
        watermark_mask: Sequence[bool] | None = None,
    ) -> DetectionResult:
        """Per-position DISC tests mixed by Fisher combination of q-values.

        Call flow::

            assign_sequence(token_ids) or round-robin assignment
              -> group each token's bit indices by position
              -> search all mapped symbols for each nonempty group
              -> select the smallest local p-value per position
              -> multiply by the symbol count for union correction
              -> combine corrected p-values with Fisher's method
              -> join recovered symbols into the full payload if detected

        Args:
            bits: list[int]: Complete flat observed 0/1 stream of length N_bits.
            token_ids: Sequence[int]: Complete flat sequence of N_tokens unsigned 32-bit
                IDs; N_bits = N_tokens * bit_length.
            bit_length: int: Positive scalar w, the fixed number of MSB-first bits per
                token. For aligned streams, N_bits = N_tokens * w.
            n_star: int: Nonnegative scalar prefix length in bits; the prefix is
                bits[:n_star].
            n_star_tokens: int: Nonnegative scalar prefix length in tokens; used by
                prefix_token_ngram.
            watermark_mask: Sequence[bool] | None: Flat mask of length N_bits aligned with
                bits; False excludes that bit from scoring. None supplies no explicit
                exclusions.

        Returns:
            DetectionResult: One immutable result. On detection, payload is in [0,
            2**(payload_bits * H)) and symbols is a tuple of H decoded integers; otherwise
            both are None. position_p_values has one float per nonempty assigned group (at
            most H). Score/count aggregate those groups; n_star is in bits.
        """
        if self.use_cabs:
            if self.context_mode in ("prefix_bit_ngram", "bit_ngram"):
                # Binary DISC schedules individual bits. Scale every CABS
                # length from real-token units to binary-token units, then
                # replay using the exact h*w-bit PRF context at each bit.
                cabs_h, effective_config = _binary_cabs_settings(
                    self.cabs_config, self.n_positions, self.context_width, bit_length
                )
                scheduler = CabsScheduler(self.prf, self.n_positions, cabs_h, effective_config)
                bit_h = self._binary_h(bit_length)
                eligibility_contexts = [
                    bits[index - bit_h : index] if index >= bit_h else None
                    for index in range(len(bits))
                ]
                start_index = n_star if self.context_mode == "prefix_bit_ngram" else 0
                assignments = scheduler.assign_sequence(
                    bits,
                    start_index=start_index,
                    eligibility_contexts=eligibility_contexts,
                )
                grouped = [[] for _ in range(self.n_positions)]
                for bit_index, position in enumerate(assignments):
                    if position is not None:
                        grouped[position].append(bit_index)
            else:
                # Token-context DISC schedules whole real tokens and uses the
                # h preceding token IDs as the exact PRF eligibility context.
                eligibility_contexts: list[Sequence[int] | None] = []
                for token_index in range(len(token_ids)):
                    eligibility_contexts.append(
                        token_ids[token_index - self.context_width : token_index]
                        if token_index >= self.context_width
                        else None
                    )
                scheduler = CabsScheduler(
                    self.prf, self.n_positions, self._cabs_h, self.cabs_config
                )
                assignments = scheduler.assign_sequence(
                    token_ids,
                    start_index=n_star_tokens if self.context_mode == "prefix_token_ngram" else 0,
                    eligibility_contexts=eligibility_contexts,
                )
                grouped = [[] for _ in range(self.n_positions)]
                for token_index, position in enumerate(assignments):
                    if position is not None:
                        start = token_index * bit_length
                        grouped[position].extend(range(start, start + bit_length))
        else:
            if self.context_mode == "prefix_token_ngram":
                start_index = n_star_tokens
            else:
                start_index = 0
            assignments = [
                None if index < start_index else (index - start_index) % self.n_positions
                for index in range(len(token_ids))
            ]
            grouped = [[] for _ in range(self.n_positions)]
            for token_index, position in enumerate(assignments):
                if position is not None:
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
                # Exhaustive symbol search is small because one position holds
                # only ``payload_bits`` bits.
                score, count = self._score_indices(
                    bits,
                    grouped[position],
                    mapped,
                    n_star=n_star,
                    tokens=token_ids,
                    bit_length=bit_length,
                    n_star_tokens=n_star_tokens,
                    watermark_mask=watermark_mask,
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
            # Correct the best symbol hypothesis before combining positions.
            position_p.append(min(1.0, self.message_count * best_local))
            local_p_values.append(best_local)
            total_score += best_score
            total_count += best_count

        # Fisher combines the independent position-level q-values into one
        # decision using the detector's single configured FPR threshold.
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
        watermark_mask: Sequence[bool] | None = None,
    ) -> DetectionResult:
        """Search payloads and prefix lengths; return the corrected test.

        For prefix modes with ``use_prefix=True``, each candidate ``n_star``
        treats ``bits[:n_star]`` as ``R``. With ``use_prefix=False``, only
        ``n_star=0`` (``R = []``) is tested. Scoring calls the same
        ``prf_uniform`` as encode. See the module call flow.

        Call flow::

            detect
              -> copy and validate the observed bits
              -> choose the single-position or multi-position path
              -> reconstruct the encoder's contexts and PRF samples
              -> compute DISC scores and tail probabilities
              -> correct for searched starts and symbols
              -> return detection status, payload, scores, and diagnostics

        The method does not mutate the caller's ``bits`` sequence. It makes a
        list copy because the scoring helpers use indexed slices and because
        detection may need to pass the same immutable observation through many
        candidate hypotheses.

        The detector never computes ``empirical_entropy`` and never uses the
        encoder's ``entropy_threshold``. For prefix modes it only hypothesizes
        candidate prefix lengths and evaluates their statistical scores.

        Args:
            bits: Sequence[int]: Flat binary sequence of length N_bits, containing only 0
                and 1 in stream order.
            n_star_candidates: Iterable[int] | None: Finite iterable of K candidate prefix
                lengths. Units are tokens for prefix_token_ngram, bits otherwise. None
                selects the mode’s default search; candidates are unnecessary without a
                prefix.
            token_ids: Sequence[int] | None: Flat sequence of N_tokens unsigned 32-bit IDs
                aligned with bits. Supply the original IDs for real-token contexts/CABS;
                when omitted on the token/position path, each bit is treated as a token.
            bit_length: int | None: Positive scalar w with N_bits = N_tokens * w. With
                token_ids, None infers N_bits // max(N_tokens, 1); without IDs on the
                token/position path, w = 1.
            watermark_mask: Sequence[bool] | None: Flat mask of length N_bits aligned with
                bits; False excludes that bit from scoring. None supplies no explicit
                exclusions.

        Returns:
            DetectionResult: One immutable object with scalar status, payload, prefix length
            in bits, p-values, score in nats, and scored-bit count (see DetectionResult
            attributes). The position path additionally returns H decoded symbols on
            detection and up to H position p-values; the single-position path leaves
            symbols=None and position_p_values=().
        """
        bits = list(bits)
        _validate_bits(bits)
        if watermark_mask is not None:
            watermark_mask = list(watermark_mask)
            if len(watermark_mask) != len(bits):
                raise ValueError("watermark_mask must have the same length as bits")
        bit_h = self._binary_h(bit_length)
        if self.use_cabs or self.n_positions > 1 or self.context_mode in (
            "token_ngram",
            "prefix_token_ngram",
        ):
            # Multi-position and token-context modes require token structure;
            # for a raw bit stream each bit is treated as one token. If the
            # encoder used selector mode, reconstruct its mask before scoring.
            if token_ids is None:
                # When raw bits were emitted in fixed-width real-token groups,
                # recover those IDs so CABS assigns one position per group.
                # The default width of one preserves the legacy one-bit-token
                # interpretation when no wider token representation is given.
                bit_length = bit_length or self.bits_per_token
                if bit_length > 1 and len(bits) % bit_length:
                    raise ValueError("bits length must be divisible by bit_length")
                token_ids = (
                    bits_to_token_ids(bits, bit_length)
                    if bit_length > 1
                    else bits
                )
            else:
                token_ids = list(token_ids)
                bit_length = bit_length or (len(bits) // max(len(token_ids), 1))
            if watermark_mask is None and self.randomize_without_prefix and not self.use_prefix:
                watermark_mask = self._randomization_mask(bits, token_ids, bit_length)
            if not self.use_prefix:
                candidates = [0]
            elif n_star_candidates is not None:
                candidates = [int(c) for c in n_star_candidates if c is not None]
            elif self.context_mode == "prefix_token_ngram":
                # Search token boundaries rather than every binary boundary.
                # Candidate k represents tokens[:k] as R and is reported in
                # bits as n_star = k * bit_length.  This gives N_tokens - 1
                # hypotheses instead of N_bits - 1 hypotheses.
                candidates = range(1, len(token_ids))
            else:
                # Prefix bit context permits every nonempty proper bit prefix.
                candidates = range(1, len(bits))

            best_result: DetectionResult | None = None
            tested_starts = 0
            for candidate in candidates:
                if self.context_mode == "prefix_token_ngram":
                    n_star_tokens = int(candidate)
                    n_star = n_star_tokens * bit_length
                else:
                    n_star_tokens = 0
                    n_star = int(candidate)
                if n_star < 0 or n_star >= len(bits) or (self.use_prefix and n_star == 0):
                    continue
                tested_starts += 1
                result = self._detect_positions(
                    bits,
                    token_ids,
                    bit_length,
                    n_star,
                    n_star_tokens,
                    watermark_mask,
                )
                if best_result is None or result.global_p_value < best_result.global_p_value:
                    best_result = result
            if best_result is not None:
                # _detect_positions corrects the symbol search at each
                # position. Correct the additional search over prefix starts
                # before applying the detector's overall FPR threshold.
                corrected_p = (
                    -math.expm1(tested_starts * math.log1p(-best_result.global_p_value))
                    if best_result.global_p_value < 1.0
                    else 1.0
                )
                detected = corrected_p <= self.fpr
                return DetectionResult(
                    detected,
                    best_result.payload if detected else None,
                    best_result.n_star,
                    best_result.local_p_value,
                    corrected_p,
                    best_result.score,
                    best_result.scored_bits,
                    best_result.symbols if detected else None,
                    best_result.position_p_values,
                )
            return DetectionResult(False, None, None, 1.0, 1.0, 0.0, 0)

        if len(bits) <= bit_h:
            # Without a complete context window there is no valid score.
            return DetectionResult(False, None, None, 1.0, 1.0, 0.0, 0)
        if watermark_mask is None and self.randomize_without_prefix and not self.use_prefix:
            watermark_mask = self._randomization_mask(bits, bit_length=1)
        if not self.use_prefix:
            n_star_candidates = [0] if n_star_candidates is None else n_star_candidates
        candidates = (
            [int(c) for c in n_star_candidates if c is not None]
            if n_star_candidates is not None
            else range(1, len(bits))
        )
        tested_starts = 0
        best: tuple[float, float, int, int | None, int | None] = (1.0, 0.0, 0, None, None)
        for n_star in candidates:
            if n_star < 0 or n_star >= len(bits) or max(n_star, bit_h) >= len(bits):
                continue
            tested_starts += 1
            for payload in range(self.message_count):
                # Keep only the most significant hypothesis by local p-value;
                # multiple-testing correction is applied after this search.
                score, count = self._score(bits, n_star, payload, watermark_mask)
                local_p = float(gammaincc(count, score)) if count else 1.0
                if local_p < best[0]:
                    best = (local_p, score, count, n_star, payload)

        local_p, score, count, n_star, mapped_payload = best
        # Correct for all payload hypotheses and, when enabled, all candidate
        # prefix starts before comparing with the requested FPR.
        per_start = min(1.0, self.message_count * local_p)
        n_start_tests = 1 if not self.use_prefix else tested_starts
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
