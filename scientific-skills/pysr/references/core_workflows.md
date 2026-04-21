# PySR Core Workflows

This is the practical workflow for most PySR tasks.

## If you only remember five things

1. Start with the smallest operator set that could plausibly express the target.
2. Treat run 1 as a cheap probe, not the final search.
3. Inspect `model.equations_`, not just `get_best()`.
4. Tighten operators and constraints before adding more runtime.
5. Compare a few Pareto-front candidates before shipping one equation.

## Table of contents
1. Standard symbolic regression workflow
2. Tuning sequence that usually works
3. Operators, constraints, and complexity
4. Losses, weights, and batching
5. Reading results and exporting equations
6. Common retuning loop
7. Warm starts and reruns
8. High-dimensional or redundant feature sets

## 1. Standard symbolic regression workflow

### Minimal pattern

```python
import numpy as np
from pysr import PySRRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 4))
y = X[:, 0] ** 2 + np.sin(X[:, 1]) - 0.5 * X[:, 2]

model = PySRRegressor(
    niterations=300,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "square"],
    maxsize=18,
    parsimony=1e-3,
)
model.fit(X, y)

print(model.equations_[["loss", "complexity", "equation"]].tail())
```

### Recommended habit

Treat the first run as a search-space probe, not the final answer.

Goals for run 1:
- verify the target is learnable at all
- check whether the right variables appear
- see whether complexity is exploding
- decide whether the operator set is too broad or too narrow

### Choosing from the Pareto front

This is the most important plain-PySR habit.

PySR does not just return one equation. It returns a frontier of equations trading off fit and complexity. In many real workflows, the best operational answer is not the minimum-loss row, but the simplest row whose loss is already good enough.

Useful selection patterns:
- `model_selection="best"`: good default when you want PySR to balance simplicity and accuracy.
- `model_selection="accuracy"`: useful when the user explicitly wants the lowest-loss available equation.
- `model_selection="score"`: useful for advanced workflows that want to rank directly by score instead of PySR's `best` thresholding.
- manual inspection of `model.equations_`: best when interpretability matters and the user is willing to choose from the frontier.

Typical workflow:

```python
cols = [c for c in ["loss", "complexity", "score", "equation"] if c in model.equations_.columns]
print(model.equations_[cols].tail(10))
```

If two candidate equations have very similar loss, usually prefer the simpler one unless the user explicitly says otherwise.

One subtlety from the implementation: the `score` column is only computed in the default log-scaled loss workflow. If `score` is absent, PySR falls back to accuracy-based selection internally, so do not expect `model_selection="best"` to behave identically under every loss setup.

## 2. Tuning sequence that usually works

This sequence mirrors the tuning notes in the PySR docs.

### Step 1, start with fewer operators than you want

Good default mindset:
- only include operators you would be comfortable seeing in the final equation
- avoid redundant operators if a simpler one already spans the same behavior

Examples:
- if `square` is available, you often do not need unrestricted `pow`
- if `sin` is available, you may not need both `sin` and `cos`
- do not add `exp`, `log`, and multiple custom operators unless the domain really calls for them

### Step 2, keep `maxsize` modest

A common failure mode is giving PySR a giant search space before it has any clue what matters.

Reasonable starting values:
- `maxsize=12` to `20` for many clean toy or tabular problems
- increase only when the Pareto front clearly wants more structure

### Step 3, use constraints early

`constraints` and `nested_constraints` are not polish, they are search guidance.

Example:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*", "pow"],
    unary_operators=["sin"],
    constraints={"pow": (9, 1)},
    nested_constraints={"sin": {"sin": 0}},
)
```

This is directly in line with the project tuning guide: strict constraints often help a lot.

### Step 4, scale budget only after structure looks sane

Useful knobs once the operator set is credible:
- `niterations`
- `populations`
- `ncycles_per_iteration`
- `weight_optimize`
- `batching`

For big cluster runs, the docs recommend increasing `ncycles_per_iteration` substantially so workers communicate less often.

## 3. Operators, constraints, and complexity

## Choosing operators

Good first operator menus:

### Operator-selection cookbook

Use the smallest operator family that matches the user's actual hypothesis.

| Situation | First operator set | What to watch |
|---|---|---|
| Mostly polynomial structure | `+`, `-`, `*`, maybe `square` | complexity blow-up from unnecessary `pow` |
| Smooth periodic behavior | `+`, `-`, `*`, maybe `/`, then `sin` | do not start with `sin` and `cos` and `tan` together |
| Growth or decay | `+`, `-`, `*`, maybe `/`, then one of `exp` or `log` | avoid adding both `exp` and `log` immediately |
| Rational-form candidate | `+`, `-`, `*`, `/` | constrain aggressively before increasing runtime |
| Peaked or localized behavior | small base set plus a custom Gaussian-like operator | verify export mappings and operator safety |

Bad default habit: throwing in every vaguely plausible operator and hoping more runtime will sort it out.

### Polynomial-ish
```python
binary_operators=["+", "-", "*"]
unary_operators=["square"]
```

### Smooth nonlinear
```python
binary_operators=["+", "-", "*", "/"]
unary_operators=["sin", "exp"]
```

### With a custom operator
```python
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["gauss(x) = exp(-x^2)"],
    extra_sympy_mappings={"gauss": lambda x: sympy.exp(-(x**2))},
)
```

The Gaussian operator is a practical choice for resonance-like or peaked fits.

If you want this operator to participate cleanly in downstream exports, also define matching JAX or Torch mappings when those export targets matter.

## Custom operator rules that matter

A custom operator should:
- work over the intended floating-point type
- avoid throwing exceptions over the real line
- return `NaN` for invalid inputs rather than crashing
- use Julia float literals like `2.5f0` rather than `2.5` inside operator definitions
- have `extra_sympy_mappings` if you want `lambda_format` and SymPy export to work
- have `extra_jax_mappings` and `extra_torch_mappings` too if JAX or Torch export matters

PySR may probe custom operators outside the observed data range during search, so “works on my training range” is not enough.

## Complexity controls

Important levers:
- `maxsize`
- `maxdepth`
- `parsimony`
- `complexity_of_operators`
- `warmup_maxsize_by`

Useful pattern when equations get too complicated too early:

```python
model = PySRRegressor(
    warmup_maxsize_by=0.5,
    complexity_of_operators={"pow": 3, "exp": 2},
)
```

## 4. Losses, weights, and batching

### Elementwise loss

Use Julia syntax and keep it elementwise:

```python
elementwise_loss="loss(prediction, target) = (prediction - target)^2"
```

An `elementwise_loss` should not loop over the full dataset.

### Common built-in loss choices

- `L2DistLoss()` is the default starting point.
- `L1DistLoss()` is often worth trying on noisy or outlier-heavy data.
- `HuberLoss(d)` can be a useful middle ground when pure squared error is too sensitive but pure absolute error is too blunt.

On harder fitting problems, it is often worth changing both the transformed target and the loss choice.

### Weights

If some points matter more, pass weights to `fit`:

```python
weights = 1.0 / (sigma ** 2)
model.fit(X, y, weights=weights)
```

If you also define a custom `elementwise_loss`, it must accept three arguments:

```python
elementwise_loss="loss(prediction, target, weight) = weight * abs(prediction - target)"
```

If you pass `weights=...` but keep a two-argument custom loss, PySR will not know how to apply the weights correctly.

If the objective really needs to look across the full dataset, use the full-objective path rather than trying to smuggle dataset-level logic into `elementwise_loss`.

### Batching

The tuning notes give a practical rule:
- for datasets bigger than about 1000 rows, either subsample or use batching

Important caveat: batching is mainly about reducing cost on larger datasets. It is not a generic recommendation for noisy data. For small noisy datasets, batching can make selection less stable and may be worse than just fitting on the full dataset.

One important current-version detail: `batching` defaults to `"auto"`, so PySR already turns batching on automatically for larger datasets.

Example:

```python
model = PySRRegressor(
    batching=True,
    batch_size=128,
)
```

If a large run is underperforming, try a smaller batch size before making the search space larger.

## 5. Reading results and exporting equations

After fitting, inspect:

```python
model.equations_
model.get_best()
model.sympy()
model.latex()
```

### What to inspect first

For most users, the best first view is the equation table itself:

```python
cols = [c for c in ["loss", "complexity", "score", "equation"] if c in model.equations_.columns]
print(model.equations_[cols])
```

Notes:
- `loss` and `complexity` are the most universal columns to reason about.
- `score` is useful when present, but do not assume it is always available or comparable across every loss setup.
- `model.get_best()` reflects the current `model_selection` strategy, not some universally correct choice.

Useful exports usually include:
- `equation`
- `sympy_format`
- `lambda_format`
- optionally JAX and Torch forms when those exports are enabled and their mappings exist

`lambda_format` is the main callable form for normal Python-side evaluation. Describing this as a “NumPy export” is fine informally, but in practice the concrete surface is the callable/lambda representation plus SymPy/JAX/Torch helpers.

### Saved artifacts

Typical outputs:
- `checkpoint.pkl`
- `hall_of_fame.csv`

Reload with:

```python
from pysr import PySRRegressor
model = PySRRegressor.from_file(run_directory="outputs/2026-01-01_120000.000")
```

If the user quit a run early, the run directory can still be useful because the CSV updates during the search and `checkpoint.pkl` may exist from the latest checkpoint.

For plain searches this reload flow is well supported. For template-based runs, treat reload as fragile: `TemplateExpressionSpec` pickled reloads can still fail.

## 6. Common retuning loop

This is a more realistic workflow than trying to get the final equation in one shot.

1. Start with a minimal operator set and modest `maxsize`.
2. Run a short search.
3. Inspect `model.equations_`.
4. Decide whether the frontier is failing because of:
   - missing operator expressivity
   - too much complexity freedom
   - too many rows for the current search budget
   - wrong loss choice
5. Tighten constraints or add one operator.
6. Restart fresh unless the setup is materially unchanged.

Example:

```python
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*"],
    unary_operators=["square"],
    maxsize=14,
)
model.fit(X, y)

# After inspection, maybe add one operator and restart fresh:
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    maxsize=14,
)
model.fit(X, y)
```

## 7. Warm starts and reruns

### Good use of `warm_start`

Use `warm_start=True` when you want more search time with essentially the same setup.

```python
model = PySRRegressor(
    warm_start=True,
    niterations=50,
    binary_operators=["+", "*"],
)
model.fit(X, y)
model.fit(X, y)  # continue search
```

### Unsafe changes before warm-starting

These changes are risky or incompatible enough that you should usually restart fresh:
- `expression_spec`
- `maxsize`
- `maxdepth`
- `binary_operators`
- `unary_operators`
- `precision`
- often also constraints and complexity rules

If in doubt, use `model.reset()` or start a fresh run.

## 8. High-dimensional or redundant feature sets

PySR can handle more than toy feature counts, but search cost still grows brutally.

Practical advice:
- if there are many candidate features, prefilter with a faster model first
- if the structure is known, encode it with a template instead of hoping unconstrained search rediscovers it
- reduce operator count before increasing runtime
- use batching or subsampling for large row counts

On highly redundant feature libraries, do feature selection with a faster model such as XGBoost or a Shapley-style analysis before PySR.

As a sanity check, 16 features is not impossible, but it is enough that careless search spaces become expensive fast.
