"""Monte Carlo experiment corresponding to Section 5 of the paper.

Instead of an LM, each next-bit probability is drawn Uniform(0, 1). That
isolates DISC's encoder/detector from any particular vocabulary.

Typical call::

    summary = run_simulation(runs=100, payload_bits=4, sequence_bits=17 * 20)
    # summary.bit_error_rate is float in [0, 1], e.g. 0.02
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .core import ContextMode, DiscDetector, DiscEncoder, MessageMapping


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate metrics over independent Bernoulli DISC trials.

    Attributes:
        runs: Number of independent trials, ``int``. Example: ``100``.
        payload_bits: Message length ``m``, ``int``. Example: ``4`` (16 messages).
        sequence_bits: Binary tokens generated per trial, ``int``.
            Paper default is ``17 * n`` for ``n`` "real" tokens, e.g.
            ``17 * 20 = 340``.
        bit_error_rate: Mean fraction of payload bits wrong, ``float`` in
            ``[0, 1]``. Missed detections count as all ``m`` bits wrong.
            Example: ``0.04`` means 4% of payload bits erred.
        message_accuracy: Fraction of trials with both detection and the
            exact payload, ``float`` in ``[0, 1]``. Example: ``0.91``.
        detection_rate: Fraction of trials with ``result.detected``,
            ``float`` in ``[0, 1]``. Example: ``0.93`` (may include wrong
            payloads, so it can exceed ``message_accuracy``).
    """

    runs: int
    payload_bits: int
    sequence_bits: int
    bit_error_rate: float
    message_accuracy: float
    detection_rate: float
    generation_seconds: float
    decoding_seconds: float


def run_simulation(
    *,
    runs: int = 1000,
    payload_bits: int = 4,
    sequence_bits: int = 17 * 20,
    context_width: int = 4,
    entropy_threshold: float = 5.0,
    fpr: float = 0.01,
    seed: int = 0,
    message_mapping: MessageMapping = "direct",
    use_prefix: bool | None = None,
    context_mode: ContextMode = "prefix_bit_ngram",
) -> SimulationSummary:
    """Simulate random Bernoulli conditionals as described in Section 5.

    Each trial: draw a random payload, encode ``sequence_bits`` binary tokens
    whose P(bit=1) values are i.i.d. Uniform(0, 1), then detect with the
    matching key. Keys are per-trial strings ``"simulation-key-{run}"``.

    Args:
        runs: Positive ``int`` trial count. Example: ``100``.
        payload_bits: ``m``, positive ``int``. Example: ``4``.
        sequence_bits: Binary tokens per trial, ``int`` strictly greater than
            ``context_width``. Example: ``340`` (= 17 bits × 20 tokens).
        context_width: n-gram length ``h`` in real tokens. For this Bernoulli
            simulation ``|V|=2`` so ``ceil(log2 |V|)=1`` and the binary
            context is ``h`` bits. Example: ``4``.
        entropy_threshold: Random-init stop in nats, ``float``. Example: ``5.0``.
        fpr: Detector false-positive target, ``float`` in ``(0, 1)``.
            Example: ``0.01``.
        seed: Master RNG seed, ``int``. Example: ``0``. Controls payloads,
            Bernoulli probabilities, and each encoder's random-init seed.
        message_mapping: ``"direct"`` or ``"gray"``. Applied to both encoder
            and detector.
        use_prefix: If False, encode/detect with ``R = []`` (no ``n_star``
            search). ``None`` follows ``context_mode``: prefix modes use R;
            non-prefix modes do not.
        context_mode: PRF context representation used by both encoder and
            detector. Raw Bernoulli bits act as one-token IDs in token modes.

    Returns:
        ``SimulationSummary`` with the six fields documented on that class.
        Example: ``SimulationSummary(runs=100, payload_bits=4,
        sequence_bits=340, bit_error_rate=0.03, message_accuracy=0.9,
        detection_rate=0.92)``.
    """

    if runs < 1 or sequence_bits <= context_width:
        raise ValueError("runs must be positive and sequence_bits must exceed context_width")
    rng = random.Random(seed)
    bit_errors = correct = detected = 0  # ints accumulated over trials
    generation_seconds = decoding_seconds = 0.0
    for run in range(runs):
        payload = rng.randrange(2**payload_bits)  # int in {0, ..., 2^m - 1}, e.g. 11
        encoder = DiscEncoder(
            f"simulation-key-{run}",  # str key unique to this trial
            payload,
            payload_bits,
            entropy_threshold=entropy_threshold,
            context_width=context_width,
            seed=rng.randrange(2**63),  # int encoder RNG seed
            message_mapping=message_mapping,
            use_prefix=use_prefix,
            context_mode=context_mode,
        )
        generation_start = time.perf_counter()
        for _ in range(sequence_bits):
            encoder.encode_bit(rng.random())  # p_one is float Uniform(0, 1)
        generation_seconds += time.perf_counter() - generation_start
        decoding_start = time.perf_counter()
        result = DiscDetector(
            f"simulation-key-{run}",
            payload_bits,
            context_width=context_width,
            fpr=fpr,
            message_mapping=message_mapping,
            use_prefix=use_prefix,
            context_mode=context_mode,
        ).detect(encoder.bits)  # bits: list[int] of length sequence_bits
        decoding_seconds += time.perf_counter() - decoding_start
        detected += int(result.detected)  # 0 or 1
        decoded = result.payload if result.payload is not None else 0  # int
        # XOR bit count: Hamming distance between true and decoded payload.
        # Missed detection is charged as m bit errors.
        errors = (payload ^ decoded).bit_count() if result.detected else payload_bits
        bit_errors += errors
        correct += int(result.detected and decoded == payload)
    return SimulationSummary(
        runs,
        payload_bits,
        sequence_bits,
        bit_errors / (runs * payload_bits),  # float BER
        correct / runs,  # float accuracy
        detected / runs,  # float detection rate
        generation_seconds,
        decoding_seconds,
    )
