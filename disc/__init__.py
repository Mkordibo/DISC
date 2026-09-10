"""Reference implementation of Distribution Interval Shift Coding (DISC).

Public API for encoding a small integer payload into a stream of binary tokens
and later detecting/decoding that payload from the bits. Optional CABS splits
the payload across positions; each position is still DISC-watermarked.
"""

from .cabs import CabsConfig, CabsScheduler
from .core import (
    DetectionResult,
    DiscDetector,
    DiscEncoder,
    HmacPrf,
    bits_to_token_ids,
    combine_p_values,
    conditional_bit_probability,
    disc_score,
    gray_decode,
    gray_encode,
    join_payload,
    split_payload,
    token_ids_to_bits,
)
from .experiment import ExperimentConfig, build_record, load_prompts, parse_key

__all__ = [
    "CabsConfig",
    "CabsScheduler",
    "DetectionResult",
    "DiscDetector",
    "DiscEncoder",
    "ExperimentConfig",
    "HmacPrf",
    "bits_to_token_ids",
    "build_record",
    "combine_p_values",
    "conditional_bit_probability",
    "disc_score",
    "gray_decode",
    "gray_encode",
    "join_payload",
    "load_prompts",
    "parse_key",
    "split_payload",
    "token_ids_to_bits",
]
