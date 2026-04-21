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
- exporting equations to SymPy, NumPy, JAX, PyTorch, or LaTeX
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
- `warm_start=True` is useful only if core search settings stay effectively the same.

### Structured searches with templates
- Reach for `TemplateExpressionSpec` when you already know the outer equation form or need shared subexpressions.
- For category-specific parameters, add the category as a column in `X` and remember Julia is 1-indexed.
- Keep template `combine` returning a single scalar. If you need multi-output behavior, encode residuals into one scalar objective.

### Losses and objectives
- `elementwise_loss` should be truly elementwise. Do not sum over rows inside it.
- For likelihood-style or signed objectives, consider `loss_scale="linear"`.
- For noisy tails or outliers, try `L1`-style losses before overengineering the operator set.

### Interpreting outputs
- `model.equations_` is the main artifact. Use it to compare loss, score, and complexity.
- Saved files usually include both `hall_of_fame...csv` and `hall_of_fame...pkl`.
- Use `PySRRegressor.from_file(...)` to inspect a saved run in a fresh process.

## Discussion-derived gotchas worth remembering

- Template categories are effectively 1-indexed. If category ids start at 0, add 1 before fitting. This came up directly in discussion #1179.
- `TemplateExpressionSpec.combine` must return one value, not a tuple. Multi-output problems need a residual-based workaround, discussed in #1174 and #1002.
- `warm_start` is brittle if you change operators, `expression_spec`, `maxsize`, `maxdepth`, or precision, per #922 and the README.
- On one machine, prefer multithreading over distributed cluster setup unless there is a clear reason otherwise. This advice recurs in #1144 and #1110.
- Template expressions on distributed workers have been a source of edge cases, especially around Slurm in #1110.
- If `hall_of_fame.csv` permissions behave strangely, write to a simple local writable directory rather than a sync folder or quirky mount, echoing #1048.

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
