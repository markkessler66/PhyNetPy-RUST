# PhyNetPy

PhyNetPy is a Python library for phylogenetic network inference and analysis. It
provides improved implementations of methods from PhyloNet, plus a framework for
building new Bayesian and simulation-based methods.

Starting with 0.8.0 the computational core is written in Rust. The Python
package `phynetpy` is the public API and has the same classes, functions, and
module layout as before; underneath, the graph structures, parsers, likelihood
kernels, and search engines run in the compiled extension `phynetpy._core`.

## Install

```bash
pip install phynetpy
```

Wheels are published for Python 3.9 and later (one `abi3` wheel per platform).
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
pip install maturin pytest numpy scipy scikit-learn matplotlib networkx biopython python-nexus newick pulp
maturin develop            # build the extension into the venv (add --release for speed)
python -m pytest -m "not slow"
```

The version lives in `python/phynetpy/_version.py` and `Cargo.toml`; keep the
two in step.
