from __future__ import annotations

import argparse
import time
from pathlib import Path

from simulation import (
    DEFAULT_M,
    DEFAULT_N,
    DEFAULT_REPETITIONS,
    PAPER_REPETITIONS,
    run_pareto_experiments,
    run_standard_experiments,
    write_result_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Monte Carlo experiments for target-aware linear regression."
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "pareto"),
        default="standard",
        help="Use 'standard' for the noise/covariance/mean regimes, or 'pareto' for runtime-accuracy results.",
    )
    parser.add_argument(
        "--regime",
        nargs="+",
        default=["all"],
        choices=("noise", "covariance", "mean", "all"),
        help="Standard-regime subset to run. Ignored in pareto mode.",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Training sample size.")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="Test sample size.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Monte Carlo repetitions per configuration. Use 1000000 for the paper setting.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed. Use -1 for an unseeded run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path to write. Defaults to results/experiment_results.csv or results/pareto_results.csv.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every this many repetitions within each configuration.",
    )
    parser.add_argument(
        "--paper-repetitions",
        action="store_true",
        help=f"Shortcut for --repetitions {PAPER_REPETITIONS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 4:
        raise ValueError("n must be larger than d + 1 for the default d=3 experiment.")
    if args.m <= 3:
        raise ValueError("m must be larger than d for exact test normalization.")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative.")

    repetitions = PAPER_REPETITIONS if args.paper_repetitions else args.repetitions
    seed = None if args.seed < 0 else args.seed
    default_name = "pareto_results.csv" if args.mode == "pareto" else "experiment_results.csv"
    output_path = args.output if args.output is not None else Path("results") / default_name

    start = time.perf_counter()
    if args.mode == "pareto":
        rows = run_pareto_experiments(
            n=args.n,
            m=args.m,
            repetitions=repetitions,
            seed=seed,
            progress_every=args.progress_every,
        )
    else:
        rows = run_standard_experiments(
            regimes=args.regime,
            n=args.n,
            m=args.m,
            repetitions=repetitions,
            seed=seed,
            progress_every=args.progress_every,
        )

    write_result_csv(rows, output_path)
    elapsed = time.perf_counter() - start
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Elapsed seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
