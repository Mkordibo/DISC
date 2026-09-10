"""Command line interface for DISC simulations.

Invoked as ``disc-simulate`` after ``pip install -e .``. Prints a JSON object
whose keys match ``SimulationSummary`` field names.

Example::

    disc-simulate --runs 100 --payload-bits 4 --real-tokens 20

prints something like::

    {
      "runs": 100,
      "payload_bits": 4,
      "sequence_bits": 340,
      "bit_error_rate": 0.03,
      "message_accuracy": 0.91,
      "detection_rate": 0.93
    }

``sequence_bits`` is ``real_tokens * bits_per_token`` (ints; paper uses 17×n).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .simulation import run_simulation


def main() -> None:
    """Parse CLI flags, run the Bernoulli simulation, print JSON to stdout.

    Flag types and defaults:

        --runs                int,   default 100     (trial count)
        --payload-bits        int,   default 4       (m; 16 messages)
        --real-tokens         int,   default 20      (n in the paper)
        --bits-per-token      int,   default 17      (binary tokens per real token)
        --context-width       int,   default 4       (h real tokens; here |V|=2 so h bits)
        --context-mode        str,   default prefix_bit_ngram
        --entropy-threshold   float, default 5.0     (nats)
        --fpr                 float, default 0.01    (false-positive target)
        --seed                int,   default 0
        --message-mapping     str,   "direct" | "gray"
        --no-prefix           flag           encode/detect with R=[]

    Returns:
        None. Side effect: writes a JSON object to stdout.
    """
    parser = argparse.ArgumentParser(description="Run the paper's DISC Bernoulli simulation")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--payload-bits", type=int, default=4)
    parser.add_argument("--real-tokens", type=int, default=20)
    parser.add_argument("--bits-per-token", type=int, default=17)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument(
        "--context-mode",
        choices=["prefix_bit_ngram", "bit_ngram", "token_ngram", "prefix_token_ngram"],
        default="prefix_bit_ngram",
        help="PRF context representation used by the encoder and decoder",
    )
    parser.add_argument("--entropy-threshold", type=float, default=5.0)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--message-mapping", choices=["direct", "gray"], default="direct")
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="watermark and detect with R=[] (no n_star search)",
    )
    args = parser.parse_args()
    summary = run_simulation(
        runs=args.runs,
        payload_bits=args.payload_bits,
        sequence_bits=args.real_tokens * args.bits_per_token,  # e.g. 20 * 17 = 340
        context_width=args.context_width,
        entropy_threshold=args.entropy_threshold,
        fpr=args.fpr,
        seed=args.seed,
        message_mapping=args.message_mapping,
        use_prefix=False if args.no_prefix else None,
        context_mode=args.context_mode,
    )
    # asdict(summary) → dict[str, int | float]; indent=2 pretty-prints JSON.
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
