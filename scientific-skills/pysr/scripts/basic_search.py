#!/usr/bin/env python3
"""Minimal runnable PySR smoke test.

Use this script to check that a PySR environment can run a small symbolic
regression search end to end. It intentionally uses synthetic data and a narrow
operator set so failures usually point to installation/runtime issues rather
than problem setup.

Run with:
    python basic_search.py
    python basic_search.py --iterations 80 --samples 400
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50, help="PySR search iterations.")
    parser.add_argument("--samples", type=int, default=300, help="Synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data.")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation added to the target.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        from pysr import PySRRegressor
    except ImportError as exc:
        print(
            "Could not import PySR dependencies. Install PySR in this environment first.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    X = rng.uniform(-2.0, 2.0, size=(args.samples, 2))
    y = X[:, 0] ** 2 - 0.5 * X[:, 1] + 1.0
    if args.noise:
        y = y + args.noise * rng.normal(size=args.samples)

    split = max(1, int(0.8 * args.samples))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = PySRRegressor(
        niterations=args.iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        maxsize=10,
        parsimony=1e-3,
        populations=4,
        population_size=20,
        progress=False,
        random_state=args.seed,
        model_selection="best",
    )

    model.fit(X_train, y_train)

    cols = [c for c in ["loss", "complexity", "score", "equation"] if c in model.equations_.columns]
    print("\nPareto front:")
    print(model.equations_[cols])

    best = model.get_best()
    print("\nSelected equation:")
    print(best.get("sympy_format", best.get("equation")))

    if len(X_val):
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        print(f"\nValidation MSE: {val_mse:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
