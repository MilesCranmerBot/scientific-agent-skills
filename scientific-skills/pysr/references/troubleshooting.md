# PySR Troubleshooting

This guide focuses on the problems that come up repeatedly in real PySR use.

## 1. Import fails before any search starts

### Symptom
- `import pysr` crashes or aborts
- `GLIBCXX` or other library mismatch messages
- container or cluster import works inconsistently

### What to try
1. Reproduce in a fresh shell with only `python -c "import pysr"`.
2. Move to a clean environment.
3. On Linux, check whether another package is loading an incompatible `libstdc++` before Julia.
4. On clusters, prefer the provided `Apptainer.def` or a known-good container.

### Why this happens
PySR brings together Python, Julia, and compiled libraries. Import failures are usually environment problems, not symbolic-regression problems.

Relevant sources: README troubleshooting section, discussion #1020.

## 2. The search is running, but it is much slower than expected

### Common causes
- too many operators
- too-large `maxsize`
- too many rows without batching or subsampling
- using distributed execution when multithreading would be simpler
- first-run Julia compilation cost being mistaken for search cost

### What to try
- cut the operator list in half
- lower `maxsize`
- try `batching=True`
- use a smaller `batch_size`
- run a short probe job first

Discussion #948 is a useful reminder that even a moderate feature count becomes painful if the search space is bloated.

## 3. Custom operator works sometimes, then crashes or yields nonsense

### Likely cause
The operator is not safe over the numeric range PySR probes.

### What to try
- make the operator total over the real line
- return typed `NaN` for invalid inputs instead of throwing an exception
- ensure the SymPy mapping matches the Julia definition if export matters

The operators doc is explicit here: PySR expects custom operators not to throw across a very wide real range.

## 4. `elementwise_loss` is behaving strangely

### Likely cause
The loss is not actually elementwise.

### Bad pattern
```julia
function loss(predictions, targets)
    total = 0.0
    for (p, t) in zip(predictions, targets)
        total += abs(p - t)
    end
    total
end
```

### Better pattern
```julia
loss(prediction, target) = abs(prediction - target)
```

Discussion #1079 highlights this confusion directly.

### Weighted custom-loss gotcha

If you pass `weights=...` to `fit`, a custom `elementwise_loss` must accept three arguments:

```julia
loss(prediction, target, weight) = weight * abs(prediction - target)
```

If the loss only accepts `(prediction, target)`, weighted runs can fail or behave differently than intended.

## 5. TemplateExpressionSpec fails or gives bizarre results

### Checklist
- `combine` must return one scalar
- category indexing for parameter arrays must start at 1
- `variable_names` must match column order in `X`
- if doing pseudo-multi-output, use a scalar residual plus dummy `y`

Relevant discussions: #1179, #1174, #1002.

## 6. Warm start is broken or resumed runs look inconsistent

### Likely cause
Core search settings changed between fits.

### Unsafe changes include
- operators
- `expression_spec`
- `maxsize`
- `maxdepth`
- precision
- often constraints or complexity settings

### What to do
- if the search definition changed materially, start a fresh model or call `reset()`
- use `from_file(...)` to inspect old runs rather than forcing them into a new search setup

Relevant discussion: #922.

## 7. Cluster or Slurm runs fail, especially with templates

### Strong default recommendation
If the user is on one machine, prefer multithreading first.

Discussion #1144 recommends sticking with multithreading unless there is a specific reason to switch.

### Template-specific caution
Discussion #1110 suggests template expression definitions on worker nodes can be a real source of distributed failures.

### What to try
- verify the template on one process first
- remove distributed settings and confirm local correctness
- only then reintroduce `procs` or cluster settings
- ensure any worker imports or Julia-side definitions exist on workers

## 8. `hall_of_fame.csv` write errors or permissions errors

### Typical causes
- synced folders
- network mounts
- unusual file locking
- weak output-directory assumptions

### What to try
- write to a plain local directory
- simplify the output path
- avoid OneDrive, Dropbox, or exotic mounts during the run

This lines up with discussion #1048.

## 9. Progress appears stalled in tmux or remote shells

### Possible interpretations
- only the printing stalled
- the filesystem sync stalled
- the search is actually wedged

### What to check
- CPU usage
- whether the hall-of-fame files are still updating
- whether a shorter local run reproduces the issue

Discussion #1067 did not end in a clean PySR-specific root cause, so treat this as an execution-environment debugging problem, not a canonical PySR feature limitation.

## 10. Equations in CSV do not match predictions or exports

### First question
Is the numerical behavior wrong, or just the string representation/export?

### Especially relevant when
- using complex numbers
- using templates
- comparing Julia behavior to MATLAB or another language with different branch conventions

Discussion #1059 shows a real example where template-plus-complex workflows could diverge between CSV strings and evaluated model behavior.

### What to do
- test `model.predict` from the saved pickle
- compare that with evaluating the exported expression independently
- reduce to a minimal reproducible example
- be careful about branch cuts in complex arithmetic across languages

## 11. PySR ignores some variables or keeps selecting redundant feature combinations

This is often not a bug.

It usually means:
- the feature library is highly redundant
- the allowed search space is too broad
- the objective does not uniquely prefer the intended structure

### What to try
- prefilter features with a faster model
- tighten constraints
- reduce operators
- move to a template if the desired structure is already known

Discussion #1116 is the clearest recent example.

## Quick triage order

When debugging, check in this order:
1. environment import health
2. data shape and column ordering
3. operator validity
4. template correctness, if used
5. output directory writability
6. whether the run is local multithreading or distributed
7. only then start suspecting deeper PySR bugs
