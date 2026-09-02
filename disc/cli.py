"""Command line interface for DISC simulations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper's DISC Bernoulli simulation")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--payload-bits", type=int, default=4)
    parser.add_argument("--real-tokens", type=int, default=20)
    parser.add_argument("--bits-per-token", type=int, default=17)
    parser.add_argument("--context-width", type=int, default=16)
    parser.add_argument("--entropy-threshold", type=float, default=5.0)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--message-mapping", choices=["direct", "gray"], default="direct")
    args = parser.parse_args()
    summary = run_simulation(
        runs=args.runs,
        payload_bits=args.payload_bits,
        sequence_bits=args.real_tokens * args.bits_per_token,
        context_width=args.context_width,
        entropy_threshold=args.entropy_threshold,
        fpr=args.fpr,
        seed=args.seed,
        message_mapping=args.message_mapping,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
