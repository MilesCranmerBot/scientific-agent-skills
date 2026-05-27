---
name: pysr
description: Use PySR whenever the user wants symbolic regression or symbolic-model discovery: equation search, distillation, structured TemplateExpressionSpec workflows, custom losses, operator/complexity control, export, and troubleshooting around Julia startup, warm starts, and cluster usage.
license: MIT license
metadata:
  skill-author: OpenClaw
---

# PySR

## Overview

PySR is a Python interface for symbolic regression backed by SymbolicRegression.jl. Use it when the goal is to find a symbolic model you can inspect, constrain, export, and rerun, not just to maximize predictive accuracy.

Prefer plain `PySRRegressor` for unconstrained discovery. Switch to `TemplateExpressionSpec` early when important structure is already known.

## When Not to Use PySR First

Do not reach for PySR first when:
- the user mainly wants the best predictive accuracy rather than an interpretable equation
- the feature set is huge and redundant, but there is no structural prior yet
- the dataset is tiny and noisy enough that discovered equations will be unstable
- the real task is feature selection, denoising, or preprocessing rather than equation discovery

In those cases, first consider feature selection, a simpler baseline model, or a more standard predictive workflow.

## Default workflow

1. Start with a deliberately small operator set that matches the actual hypothesis class.
2. Run a short probe before spending a large search budget.
3. Inspect `model.equations_`, not just `get_best()`.
4. If equations are overly complex, tighten constraints before adding runtime.
5. If structure is partly known, switch to `TemplateExpressionSpec` early.
6. Only warm-start when the search definition is materially unchanged.

Unsuccessful PySR runs often come from an overly broad or poorly matched search space, not from too little compute.

## Task Router

| User need | Open first |
|---|---|
| Standard symbolic regression on tabular data | `references/core_workflows.md` |
| Known structure, shared parameters, or structured search | `references/template_expressions.md` |
| Import, Julia, HPC, container, or startup issues | `references/installation_and_environment.md` |
| Stalled run, bad operators, warm-start confusion, output mismatch, or file issues | `references/troubleshooting.md` |

## Default probe template

Use this as the first real run. Then inspect `model.equations_` and retune from there.

Start with the smallest operator set that could plausibly express the target.
Avoid adding `sin`, `cos`, `exp`, `log`, `pow`, `/`, and custom operators unless the domain suggests them.

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
cols = [c for c in ["loss", "complexity", "equation"] if c in model.equations_.columns]
print(model.equations_[cols].tail())
```

## Pareto Front and Model Selection

PySR does not really produce one answer. It produces an equation frontier.

Default rule:
- treat `model.equations_` as the product
- use `model_selection="best"` as a default, not as a substitute for inspection
- if interpretability matters, manually inspect the tradeoff between loss and complexity

Do not automatically choose the minimum-loss row when a slightly simpler equation is nearly as good.

## Practical Guidance

### Standard search
- Prefer a short operator list. Too many operators usually slow search more than they help.
- Start with `+`, `-`, `*`, maybe `/`, then add domain operators one by one.
- Do not begin with a broad operator menu unless the user explicitly wants exploratory search over many function families.
- `batching` defaults to `"auto"`, which already turns batching on for larger datasets.
- Treat batching mainly as a row-count tool, not a noise-handling tool.
- For small noisy datasets, batching can make model selection less stable and is often not the right default.
- For low-dimensional clean problems, subsampling often works well.
- Treat `model.equations_` as the real result. The minimum-loss row is not automatically the best operational choice if a slightly simpler equation is nearly as good.
- `model_selection="best"` is a good default, but inspect manually when interpretability matters.
- `warm_start=True` is useful only if core search settings stay effectively the same.

### Structured searches with templates
- Reach for `TemplateExpressionSpec` when you already know the outer equation form or need shared subexpressions.
- Treat templates as a first-class PySR workflow when structure is partly known.
- For category-specific parameters, add the category as a column in `X` and remember Julia is 1-indexed.
- Keep template `combine` returning a single scalar. If you need multi-output behavior, encode residuals into one scalar objective.

### Losses and objectives
- `elementwise_loss` should be truly elementwise. Do not sum over rows inside it.
- If you pass `weights=...` to `fit`, a custom `elementwise_loss` must accept three arguments: `(prediction, target, weight)`.
- Consider `loss_scale="linear"` when the custom loss can be zero or negative, or when using likelihood-style objectives.
- For noisy tails or outliers, try `L1DistLoss()` before expanding the operator set.

### Interpreting outputs
- `model.equations_` is the main artifact. Use it to compare loss and complexity, and use `score` too when that column is present.
- Saved outputs are organized in a run directory, usually including `checkpoint.pkl` plus `hall_of_fame.csv`.
- Use `PySRRegressor.from_file(run_directory=...)` to inspect ordinary saved runs in a fresh process. Template-based runs need extra care because reload support depends on dynamic expression-spec objects.

## Safe first checks before a long run

Before scaling budget, verify:
- feature count is actually modest enough for symbolic regression
- operators match the domain
- `maxsize` is not wildly larger than needed
- output path is writable
- search mode is multithreading unless distributed execution is truly required

## Key gotchas worth remembering

- The minimum-loss row is often not the best final answer. Inspect the Pareto front and prefer a simpler equation when loss is close.
- Template categories are effectively 1-indexed. If category ids start at 0, add 1 before fitting.
- If custom `elementwise_loss` is used together with `weights`, the loss must accept a third `weight` argument.
- `warm_start` is safest only when the search definition is materially unchanged. Operator changes are explicitly unsafe, and changing template/size/depth/precision settings should be treated as suspect.

## References to open next

- `references/core_workflows.md`: default PySR workflow, operator/constraint advice, result inspection, warm starts
- `references/template_expressions.md`: structured equations, category-specific parameters, shared subexpressions, multi-output workarounds
- `references/installation_and_environment.md`: install choices, Julia startup, container and cluster issues
- `references/troubleshooting.md`: bad operators, startup crashes, output-file quirks, warm-start confusion, template pitfalls

## Optional runnable examples

### `scripts/basic_search.py`
Safe-first-run PySR regression example with a held-out split, Pareto-front inspection, and cleaner scripted output. If `import pysr` is not already working, read the environment note first.

### `scripts/parametric_template.py`
Template-based example with category-specific parameters, explicit learned-vs-true parameter reporting, and the important 1-indexed category handling. It is heavier than `basic_search.py`.

## References

- PySR README: `/root/.openclaw/workspace/PySR/README.md`
- PySR tuning notes: `/root/.openclaw/workspace/PySR/docs/src/tuning.md`
- PySR examples: `/root/.openclaw/workspace/PySR/docs/src/examples.md`
- Template implementation: `/root/.openclaw/workspace/PySR/pysr/expression_specs.py`
