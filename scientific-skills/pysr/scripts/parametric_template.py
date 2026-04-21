"""PySR template example with category-specific parameters.

Important detail: category indices used to index template parameters must start at 1,
not 0, because Julia arrays are 1-indexed.

Run with:
    python parametric_template.py
"""

from __future__ import annotations

import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec


def make_dataset(n: int = 900, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3.0, 3.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    category = rng.integers(0, 3, size=n)

    scale = np.array([0.8, 1.6, -0.5])
    offset = np.array([0.2, -0.7, 1.1])
    y = scale[category] * np.sin(x1) + offset[category]
    y += 0.01 * rng.normal(size=n)

    category_for_julia = category + 1
    X = np.column_stack([x1, x2, category_for_julia])
    return X, y, category


def main() -> None:
    X, y, category = make_dataset()

    spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x1", "x2", "category"],
        parameters={"p_scale": 3, "p_offset": 3},
        combine="f(x1, x2, p_scale[category], p_offset[category])",
    )

    model = PySRRegressor(
        expression_spec=spec,
        niterations=250,
        populations=10,
        population_size=40,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin"],
        maxsize=14,
        parsimony=1e-3,
    )

    model.fit(X, y)

    print("\nBest row:")
    print(model.get_best())

    print("\nUnique zero-based categories in original data:", np.unique(category))
    print("Remember: the fitted template indexed parameters using category + 1.")


if __name__ == "__main__":
    main()
