"""PySR template example with category-specific parameters.

Important detail: category indices used to index template parameters must start at 1,
not 0, because Julia arrays are 1-indexed.

This is meant to be run in an environment where `import pysr` already works.
If you are running from a local PySR checkout, a practical pattern is:
    uv run --directory /path/to/PySR python parametric_template.py

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

    coef_sin = np.array([0.8, 1.6, -0.5])
    coef_x2 = np.array([0.2, -0.7, 1.1])
    y = coef_sin[category] * np.sin(x1) + coef_x2[category] * x2
    y += 0.01 * rng.normal(size=n)

    category_for_julia = category + 1
    X = np.column_stack([x1, x2, category_for_julia])
    return X, y, category, coef_sin, coef_x2


def main() -> None:
    X, y, category, coef_sin_true, coef_x2_true = make_dataset()

    spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x1", "x2", "category"],
        parameters={"p_sin": 3, "p_x2": 3},
        combine="f(x1, x2, p_sin[category], p_x2[category])",
    )

    model = PySRRegressor(
        expression_spec=spec,
        niterations=180,
        populations=10,
        population_size=40,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin"],
        maxsize=14,
        parsimony=1e-3,
        progress=False,
        random_state=0,
    )

    model.fit(X, y)

    best = model.get_best()

    print("\nBest row:")
    print(best)

    print("\nTemplate placeholders in the discovered equation:")
    print("#1 -> x1, #2 -> x2, #3 -> p_sin[category], #4 -> p_x2[category]")

    print("\nGround truth vs learned category parameters:")
    if "equation" in best.index:
        print("Equation:")
        print(best["equation"])
    if "julia_expression" in best.index:
        learned = str(best["julia_expression"])
        for label, truth in [("p_sin", coef_sin_true), ("p_x2", coef_x2_true)]:
            print(f"{label} true:    {truth}")
            marker = f"{label} = ["
            if marker in learned:
                start = learned.index(marker) + len(marker)
                end = learned.index("]", start)
                vals = np.fromstring(learned[start:end], sep=",")
                print(f"{label} learned: {vals}")

    print("\nUnique zero-based categories in original data:", np.unique(category))
    print("Remember: the fitted template indexed parameters using category + 1.")


if __name__ == "__main__":
    main()
