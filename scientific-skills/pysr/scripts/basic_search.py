"""Minimal PySR regression example.

Run with:
    python basic_search.py
"""

from __future__ import annotations

import numpy as np

from pysr import PySRRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(600, 3))
    y = X[:, 0] ** 2 + 1.5 * np.sin(X[:, 1]) - 0.25 * X[:, 2]
    y += 0.02 * rng.normal(size=y.shape)

    model = PySRRegressor(
        niterations=200,
        populations=8,
        population_size=33,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "square"],
        maxsize=16,
        parsimony=1e-3,
        batching=False,
        model_selection="best",
    )

    model.fit(X, y)

    print("\nBest row:")
    print(model.get_best())

    print("\nPareto front tail:")
    print(model.equations_[["loss", "complexity", "equation"]].tail())

    preds = model.predict(X[:5])
    print("\nFirst five predictions:")
    print(preds)


if __name__ == "__main__":
    main()
