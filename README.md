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

## Performance notes

MP-allop (`infer(gts, model=Allopolyploid(...), criterion=MDC())`) on DEFJ
scenario J (14 species, 7 tetraploid, 3 reticulations), Apple Silicon, one
inference per replicate with the default search budget:

| Gene trees | PhyNetPy 0.7.0 (PuLP/CBC, 10 maps) | phynetpy-rs exact (all 128 maps) | phynetpy-rs sampled (10 maps) |
| --- | --- | --- | --- |
| 10 (10 replicates) | 13.5 s | 2.5 s | 0.40 s |
| 100 (3 replicates) | 111 s | 26 s | 3.4 s |

The exact mode is the default: every valid allele map is scored when a gene
tree has at most 512 of them, so the extra-lineage count is the true
minimum (0.7.0's ten solver-chosen maps were exact only for some networks).
`allele_map_set_ilp(..., max_solutions=10)` selects the sampled mode.

Accuracy on the same runs, with every inferred network rescored over all
maps and compared to the true network (mean ± standard error):

| Gene trees | Mode | Extra lineages | mu-distance | Hardwired cluster distance |
| --- | --- | --- | --- | --- |
| 10 (n=30) | PhyNetPy 0.7.0 | 69.1 ± 5.7 | 12.3 ± 1.2 | 7.8 ± 0.9 |
| 10 (n=30) | phynetpy-rs exact | 58.4 ± 5.0 | 10.7 ± 0.9 | 6.3 ± 0.8 |
| 10 (n=30) | phynetpy-rs sampled | 60.3 ± 4.7 | 11.1 ± 1.1 | 6.9 ± 0.9 |
| 100 (n=6) | PhyNetPy 0.7.0 | 537 ± 54 | 10.3 ± 1.3 | 5.5 ± 1.1 |
| 100 (n=6) | phynetpy-rs exact | 396 ± 52 | 7.0 ± 1.1 | 3.2 ± 1.2 |
| 100 (n=6) | phynetpy-rs sampled | 680 ± 62 | 12.7 ± 1.3 | 8.0 ± 0.9 |

The true network scores 14 to 35 (10 trees) and 219 to 264 (100 trees) extra
lineages on these replicates; the search is a hill climber and is not seed
reproducible, hence the spread.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest scipy numpy matplotlib networkx biopython python-nexus
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
