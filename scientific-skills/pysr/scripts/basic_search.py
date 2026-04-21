"""Safe first-run PySR regression example.

This is meant to be run in an environment where `import pysr` already works.
If you are running from a PySR checkout, a practical pattern is:
    uv run --directory /path/to/PySR python3 basic_search.py

Run with:
    python3 basic_search.py
"""

from __future__ import annotations

import numpy as np

from pysr import PySRRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(600, 3))
    y = X[:, 0] ** 2 + 1.5 * np.sin(X[:, 1]) - 0.25 * X[:, 2]
    y += 0.02 * rng.normal(size=y.shape)

    n_train = 500
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model = PySRRegressor(
        niterations=200,
        populations=8,
        population_size=33,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "square"],
        maxsize=16,
        parsimony=1e-3,
        batching=False,
        progress=False,
        random_state=0,
        model_selection="best",
    )

    model.fit(X_train, y_train)

    print("\nBest row:")
    print(model.get_best())

    print("\nPareto front tail:")
    cols = [c for c in ["loss", "complexity", "score", "equation"] if c in model.equations_.columns]
    print(model.equations_[cols].tail())
    print("\nTip: do not blindly ship the minimum-loss row if a slightly simpler equation is nearly as good.")

    best = model.get_best()
    if "sympy_format" in best.index:
        print("\nBest equation (SymPy):")
        print(best["sympy_format"])

    preds = model.predict(X_val[:5])
    print("\nFirst five validation predictions:")
    print(preds)

    val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
    print(f"\nValidation MSE of selected equation: {val_mse:.6f}")


if __name__ == "__main__":
    main()
