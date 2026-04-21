# PySR Installation and Environment

This note covers the parts of PySR setup that actually affect whether a run starts and stays healthy.

## Recommended install paths

### Option 1, pip or uv

```bash
uv pip install pysr
# or
pip install pysr
```

PySR installs Julia-side dependencies on first import through `juliacall`.
If running from a PySR checkout, a practical pattern is often `uv run --directory /path/to/PySR python3 script.py`.

### Option 2, conda-forge

```bash
conda install -c conda-forge pysr
```

PySR is compatible with the usual Python environment managers. Use whichever environment workflow the user already trusts.

### Option 3, containerized on clusters

For HPC or locked-down systems, prefer a tested container recipe over hand-fixing libraries. The PySR repo includes:
- `Apptainer.def`
- `Dockerfile`

In practice, import problems on Singularity-like systems are often easiest to solve by starting from the provided definition file.

## What happens on first import

On first `import pysr`, PySR may:
- resolve Julia
- install Julia packages
- precompile code
- take noticeably longer than later imports

This is normal. Do not judge runtime from the first import alone.
First import and first real search may spend noticeable time provisioning and precompiling Julia packages.

## Environment assumptions that help

- Use a clean Python environment.
- Avoid multiple conflicting Julia installations on `PATH` if possible.
- Prefer local writable storage for run outputs.
- On clusters, keep the Julia depot on a real local or shared filesystem with normal file locking semantics.

## Julia version and backend notes

The checked-in `pysr/juliapkg.json` pins Julia `1.10.x` and a SymbolicRegression.jl backend revision. For users, the main practical point is simpler: use a normal supported PySR install and avoid mixing random Julia environments unless you know why.

## Common install commands to verify quickly

```bash
python3 -c "import pysr; print(pysr.__version__)"
```

If that succeeds, most environment pain is already behind you.

## Common environment failures

### 1. `GLIBCXX_... not found` or similar hard import crash

The PySR README calls out a common `libstdc++` mismatch, usually because another dependency loads the wrong C++ runtime before Julia does.

Potential fix pattern:

```bash
export LD_LIBRARY_PATH=/path/to/julia/lib/julia:$LD_LIBRARY_PATH
```

Only do this with the correct Julia library directory for that machine.

### 2. HPC or container import aborts

Start from the provided Apptainer definition instead of assembling the environment by hand.

### 3. Output file permission errors

If `hall_of_fame.csv` fails intermittently, first move the run to a simple local directory. Avoid cloud-synced folders, unusual network mounts, and aggressively scanned directories.

## Jupyter, IPython, tmux

PySR works in notebooks, but IPython is usually smoother for real searches.

Why many users prefer IPython or a plain Python process:
- better live printing
- easier interrupt behavior
- `q` + Enter early-stop support in terminal workflows

The tuning guide explicitly recommends IPython over Jupyter for long runs.

`tmux` itself is fine, but if a user reports stalled visible progress in tmux, treat that as a runtime troubleshooting problem, not proof that tmux is unsupported.

## Single-node vs cluster guidance

For most users, start with single-node multithreading.

One naming wrinkle: current PySR uses `parallelism=...` style configuration, while older docs, code snippets, and discussion threads may talk in terms of `multithreading`, `multiprocessing`, and `procs`. Read older advice through that translation layer.

Only move to distributed execution if:
- the dataset or search budget truly requires it, and
- the user is comfortable debugging Julia worker environments

Stick with multithreading unless there is a specific reason to switch.

## Minimal environment checklist

Before a serious run, verify:
- `import pysr` works in a fresh shell
- output directory is writable
- the user is not on a strange synced folder
- category/template code runs on one process first
- the operator set is small enough that slow startup is not being confused with an environment problem
