# PySR Template Expressions

`TemplateExpressionSpec` is the main tool for telling PySR, "search inside this structure, not everywhere."

It is central to many real user questions, especially:
- shared equation forms across categories
- shared subexpressions
- custom residual constructions
- derivative-based objectives
- pseudo-multi-output setups

## When to use a template

Reach for a template when one of these is true:
- you know the outer form of the equation already
- different terms should use different subsets of variables
- multiple categories share one symbolic form but differ in parameters
- you need to compute residuals from several latent subexpressions
- unconstrained PySR keeps rediscovering the wrong algebraic family

If none of those apply, plain `PySRRegressor` is usually simpler.

## Basic template pattern

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

X = np.random.randn(1000, 3)
y = np.sin(X[:, 0] + X[:, 1]) + X[:, 2] ** 2

spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
    combine="sin(f(x1, x2)) + g(x3)",
)

model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin"],
    maxsize=12,
)
model.fit(X, y)
```

## Category-specific parameters with a shared equation form

This is one of the most important template workflows because it directly addresses repeated user questions like #1179 and older threads like #530 and #1091.

### Example

Suppose all categories share the same functional form, but each category gets its own offset and scale.

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

rng = np.random.default_rng(0)
n = 600
x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-1, 1, n)
category = rng.integers(0, 3, size=n)

scale = np.array([0.8, 1.5, -0.4])
offset = np.array([0.2, 1.0, -1.2])
y = scale[category] * np.sin(x1) + offset[category]

category_p1 = category + 1  # important: Julia indexing
X = np.column_stack([x1, x2, category_p1])

spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p_scale": 3, "p_offset": 3},
    combine="f(x1, x2, p_scale[category], p_offset[category])",
)

model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin"],
    maxsize=12,
    niterations=200,
)
model.fit(X, y)
```

### Gotcha, categories are 1-indexed

This is easy to miss. Julia arrays start at 1, so category labels used to index template parameters must also start at 1.

Discussion #1179 explicitly caught a docs mistake here.

## Shared subexpressions

Templates are also good when several outputs or terms should reuse one hidden component.

Typical idea:
- create a `shared(...)` expression
- create one or more task-specific expressions
- build the final scalar residual in `combine`

This is much more efficient than hoping unconstrained symbolic regression rediscovers the shared algebra on its own.

## Multi-output workaround

`combine` must return a single scalar, not a tuple.

That means this is wrong:

```python
combine="(f1(x1, x2), f2(x1, x2))"
```

This issue appears directly in discussion #1174.

### Correct pattern

Encode the residual you care about as a scalar inside the template, then fit against a dummy target.

```python
spec = TemplateExpressionSpec(
    expressions=["f1", "f2", "shared"],
    variable_names=["x1", "x2", "y1", "y2"],
    combine="""
        s = shared(x1, x2)
        y1_hat = s + f1(x1, x2)
        y2_hat = s + f2(x1, x2)
        abs2(y1 - y1_hat) + abs2(y2 - y2_hat)
    """,
)

model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    elementwise_loss="(pred, target) -> pred",
)
```

Then pass a dummy `y`, often zeros.

This is the stable-v1 way to think about multi-output-like template problems.

## Derivatives inside templates

PySR supports a differential operator `D` in templates.

Example idea:

```python
spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x"],
    combine="df = D(f, 1); df(x)",
)
```

This is useful when the objective is naturally written in terms of derivatives, integral matching, or physics constraints. It is relevant to threads like #1174 and #1036.

## Template debugging checklist

If a template run misbehaves, check:
- `combine` returns one scalar
- all variable names in `combine` appear in `variable_names`
- category indices start at 1 if they index parameter arrays
- parameter lengths match the number of categories
- the feature matrix column order matches `variable_names`
- custom operators used inside the template are valid Julia code

## Template export and reload caveats

Template workflows are powerful, but they have had more edge cases than plain searches in discussions, including:
- loading saved template models
- distributed execution with templates
- complex-domain template exports

Relevant discussions include #1179, #1110, #1059, #1011, and #916.

Practical recommendation:
- validate the workflow on one local process first
- only then scale it up or add cluster/distributed complexity

## When not to use templates

Do not use a template just because the feature count is high.

A template helps when you know structure. If the real issue is too many redundant features, first reduce features or operators. Discussion #1116 is closer to a feature-selection problem than a template problem.
