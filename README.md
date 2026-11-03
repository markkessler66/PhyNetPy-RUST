# PhyNetPy

PhyNetPy is a Python library for phylogenetic network inference and analysis. It
provides improved implementations of methods from PhyloNet, plus a framework for
building new Bayesian and simulation-based methods.

This repository is the Rust-core build of PhyNetPy, published on PyPI as
**`phynetpy-rs`**. The Python package it installs is still named `phynetpy`
and has the same classes, functions and module layout as the pure-Python
[PhyNetPy](https://github.com/NakhlehLab/PhyNetPy) 0.7.0; underneath, the
graph structures, parsers, likelihood kernels, simulators and search operators
run in the compiled extension `phynetpy._core`. Existing scripts work
unchanged. Because both distributions provide the `phynetpy` package, install
one or the other in a given environment, not both.

## Install

```bash
pip install phynetpy-rs
```

Wheels are published for Python 3.9 and later (one `abi3` wheel per platform
for Linux x86_64/aarch64, macOS Intel/Apple Silicon and Windows x64).
Installing from source needs a Rust toolchain (`rustup`), after which

```bash
pip install .
```

builds the extension with [maturin](https://www.maturin.rs).

## Quick start

```python
from phynetpy.infer import infer, score
from phynetpy.data import GeneTrees
from phynetpy.models import MSC
from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

gts = GeneTrees.from_file("gene_trees.nex", {"A": ["A1", "A2"], "B": ["B1"]})

result = infer(gts, model=MSC(), criterion=PseudoLikelihood())
print(result.best, result.score)

log_lik = score(result.best, gts, model=MSC(), criterion=Likelihood())
```

See `README.upstream.md` for the full method table, the three-axis design of
`infer` and `score`, and the DIMPLE pipeline. Nothing about that surface
changed in the port.

## Layout

| Path | What it holds |
| --- | --- |
| `python/phynetpy/` | The `phynetpy` package: the public API, thin wrappers over the core. |
| `src/` | The Rust crate behind `phynetpy._core`. |
| `src/graph/` | `Node`, `Edge`, `Network`, transactions, graph algorithms, Newick parsing. |
| `tests/` | Test suite (`pytest`; `-m "not slow"` skips long MCMC recovery runs). |
| `examples/` | Runnable end-to-end workflows. |
| `Guides/` | Installation, I/O, and validation guides. |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest scipy numpy matplotlib networkx biopython python-nexus pulp
maturin develop            # build the extension into the venv (add --release for speed)
python -m pytest -m "not slow"
```

The version lives in `python/phynetpy/_version.py` and `Cargo.toml`; keep the
two in step. `docs/PORTING.md` describes how the Rust core and the Python
facades fit together.

## Releasing

CI (`.github/workflows/release.yml`) builds wheels for Linux (x86_64,
aarch64), macOS (Intel, Apple Silicon) and Windows (x64) plus an sdist on
every push, and runs the test suite against each wheel. Pushing a version
tag publishes those artifacts to PyPI as `phynetpy-rs` through trusted
publishing:

```bash
git tag v0.8.0
git push origin v0.8.0
```
