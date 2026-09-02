"""Reproduce the payload BER sweep described in Section 5 / Figure 5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from disc.simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--payload-bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--real-tokens", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--bits-per-token", type=int, default=17)
    parser.add_argument("--context-width", type=int, default=16)
    parser.add_argument("--entropy-threshold", type=float, default=5.0)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--message-mapping", choices=["direct", "gray"], default="direct")
    parser.add_argument("--output", type=Path, default=Path("output/figure5.csv"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows = []
    for payload_bits in args.payload_bits:
        for real_tokens in args.real_tokens:
            summary = run_simulation(
                runs=args.runs,
                payload_bits=payload_bits,
                sequence_bits=real_tokens * args.bits_per_token,
                context_width=args.context_width,
                entropy_threshold=args.entropy_threshold,
                fpr=args.fpr,
                seed=args.seed + 100_000 * payload_bits + real_tokens,
                message_mapping=args.message_mapping,
            )
            row = {
                "payload_bits": payload_bits,
                "real_tokens": real_tokens,
                "binary_tokens": summary.sequence_bits,
                "runs": summary.runs,
                "bit_error_rate": summary.bit_error_rate,
                "message_accuracy": summary.message_accuracy,
                "detection_rate": summary.detection_rate,
                "message_mapping": args.message_mapping,
            }
            rows.append(row)
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(
                "Install plotting support with: pip install -e '.[experiments]'"
            ) from exc
        for payload_bits in args.payload_bits:
            selected = [row for row in rows if row["payload_bits"] == payload_bits]
            floor = 0.5 / args.runs / payload_bits
            plt.semilogy(
                [row["real_tokens"] for row in selected],
                [max(row["bit_error_rate"], floor) for row in selected],
                marker="o",
                label=f"m={payload_bits}",
            )
        plt.xlabel("Number of real tokens")
        plt.ylabel("Bit error rate")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)


if __name__ == "__main__":
    main()
