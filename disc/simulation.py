"""Monte Carlo experiment corresponding to Section 5 of the paper."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .core import DiscDetector, DiscEncoder, MessageMapping


@dataclass(frozen=True)
class SimulationSummary:
    runs: int
    payload_bits: int
    sequence_bits: int
    bit_error_rate: float
    message_accuracy: float
    detection_rate: float


def run_simulation(
    *,
    runs: int = 1000,
    payload_bits: int = 4,
    sequence_bits: int = 17 * 20,
    context_width: int = 16,
    entropy_threshold: float = 5.0,
    fpr: float = 0.01,
    seed: int = 0,
    message_mapping: MessageMapping = "direct",
) -> SimulationSummary:
    """Simulate random Bernoulli conditionals as described in Section 5."""

    if runs < 1 or sequence_bits <= context_width:
        raise ValueError("runs must be positive and sequence_bits must exceed context_width")
    rng = random.Random(seed)
    bit_errors = correct = detected = 0
    for run in range(runs):
        payload = rng.randrange(2**payload_bits)
        encoder = DiscEncoder(
            f"simulation-key-{run}",
            payload,
            payload_bits,
            entropy_threshold=entropy_threshold,
            context_width=context_width,
            seed=rng.randrange(2**63),
            message_mapping=message_mapping,
        )
        for _ in range(sequence_bits):
            encoder.encode_bit(rng.random())
        result = DiscDetector(
            f"simulation-key-{run}",
            payload_bits,
            context_width=context_width,
            fpr=fpr,
            message_mapping=message_mapping,
        ).detect(encoder.bits)
        detected += int(result.detected)
        decoded = result.payload if result.payload is not None else 0
        errors = (payload ^ decoded).bit_count() if result.detected else payload_bits
        bit_errors += errors
        correct += int(result.detected and decoded == payload)
    return SimulationSummary(
        runs,
        payload_bits,
        sequence_bits,
        bit_errors / (runs * payload_bits),
        correct / runs,
        detected / runs,
    )
