---
name: pysr
description: Symbolic regression with PySR for discovering compact analytical equations from tabular data. Use for standard PySRRegressor workflows, constraining operator sets, tuning search complexity, structured TemplateExpressionSpec searches, custom losses, exporting equations, and practical troubleshooting around Julia startup, warm starts, and cluster usage.
license: MIT license
metadata:
  skill-author: OpenClaw
---

# PySR

## Overview

PySR is a Python interface for symbolic regression backed by SymbolicRegression.jl. Use it when the goal is not just prediction, but a compact equation you can inspect, export, and rerun.

This skill is written for stable PySR v1 style workflows first. Prefer the plain `PySRRegressor` path unless the user clearly needs a structured template, custom objective, units, or cluster setup.

## When to Use This Skill

Use this skill when tasks involve:
- discovering an interpretable equation from tabular `X, y`
- controlling allowed operators and equation complexity
- comparing simpler vs better-fitting equations on the Pareto front
- building structured expressions with `TemplateExpressionSpec`
- category-specific parameters with a shared equation form
- custom elementwise or global losses in Julia syntax
- exporting callable, SymPy, JAX, PyTorch, or LaTeX forms of equations
- resuming runs from `hall_of_fame` files or `warm_start`
- troubleshooting Julia import, cluster, or file-output issues

## Quick Start

```python
import numpy as np
from pysr import PySRRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = X[:, 0] ** 2 + 2.5 * np.sin(X[:, 1]) - 0.3

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "square"],
    maxsize=18,
    parsimony=1e-3,
    model_selection="best",
)

model.fit(X, y)
print(model)
print(model.get_best())
```

## Default Operating Pattern

1. Start with a small, non-redundant operator set.
2. Keep the first run cheap enough to iterate on quickly.
3. Inspect `model.equations_`, not just `model.predict`.
4. Tighten `constraints`, `nested_constraints`, and `maxsize` before adding more search budget.
5. Only move to `TemplateExpressionSpec` when plain symbolic regression is leaving obvious structure on the table.

## Practical Guidance

### Standard search
- Prefer a short operator list. PySR discussions repeatedly show that too many operators slow search more than they help.
- Start with `+`, `-`, `*`, maybe `/`, then add domain operators one by one.
- Use `batching=True` for larger or noisier datasets. For low-dimensional clean problems, subsampling often works well.
- Treat `model.equations_` as the real result. The minimum-loss row is not automatically the best operational choice if a slightly simpler equation is nearly as good.
- `model_selection="best"` is a good default, but it is still worth manually inspecting the Pareto front when the user cares about interpretability.
- `warm_start=True` is useful only if core search settings stay effectively the same.

### Structured searches with templates
- Reach for `TemplateExpressionSpec` when you already know the outer equation form or need shared subexpressions.
- For category-specific parameters, add the category as a column in `X` and remember Julia is 1-indexed.
- Keep template `combine` returning a single scalar. If you need multi-output behavior, encode residuals into one scalar objective.

### Losses and objectives
- `elementwise_loss` should be truly elementwise. Do not sum over rows inside it.
- If you pass `weights=...` to `fit`, a custom `elementwise_loss` must accept three arguments: `(prediction, target, weight)`.
- For likelihood-style or signed objectives, consider `loss_scale="linear"`.
- For noisy tails or outliers, try `L1`-style losses before overengineering the operator set.

### Interpreting outputs
- `model.equations_` is the main artifact. Use it to compare loss and complexity, and use `score` too when that column is present.
- Saved files usually include both `hall_of_fame...csv` and `hall_of_fame...pkl`.
- Use `PySRRegressor.from_file(...)` to inspect a saved run in a fresh process.

## Discussion-derived gotchas worth remembering

- The minimum-loss row is often not the best final answer. Inspect the Pareto front and prefer a simpler equation when loss is close.
- Template categories are effectively 1-indexed. If category ids start at 0, add 1 before fitting.
- If custom `elementwise_loss` is used together with `weights`, the loss must accept a third `weight` argument.
- `warm_start` is brittle if you change operators, `expression_spec`, `maxsize`, `maxdepth`, or precision. When in doubt, reset or start fresh.

## Bundled resources

### `references/installation_and_environment.md`
Use for install choices, Julia startup behavior, cluster/container notes, and environment-level import failures.

### `references/core_workflows.md`
Use for the main PySR workflow, tuning sequence, operator/constraint advice, exporting equations, and warm-start habits.

### `references/template_expressions.md`
Use when the user needs structured equations, category-specific parameters, shared subexpressions, derivatives, or multi-output workarounds.

### `references/troubleshooting.md`
Use for startup crashes, HPC issues, output-file quirks, invalid custom operators, warm-start confusion, and template-specific pitfalls.

### `scripts/basic_search.py`
Minimal end-to-end PySR regression example for a normal tabular problem.

### `scripts/parametric_template.py`
Template-based example with category-specific parameters and the important 1-indexed category handling.

## First things to check before a long run

```python
print(X.shape, y.shape)
print(model)
```

Then verify:
- feature count is actually modest enough for symbolic regression
- operators match the domain
- `maxsize` is not wildly larger than needed
- output path is writable
- search mode is multithreading unless distributed execution is truly required

## References

- PySR README: `/root/.openclaw/workspace/PySR/README.md`
- PySR tuning notes: `/root/.openclaw/workspace/PySR/docs/src/tuning.md`
- PySR examples: `/root/.openclaw/workspace/PySR/docs/src/examples.md`
- Template implementation: `/root/.openclaw/workspace/PySR/pysr/expression_specs.py`
