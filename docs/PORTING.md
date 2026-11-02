# Porting PhyNetPy modules to Rust

This repository is the Rust port of PhyNetPy. The public Python API must not
change; computation moves into the extension module `phynetpy._core`
(crate `phynetpy_core`, PyO3 0.30, `abi3-py39`). This note is the working
contract for every module port.

## Layout

| Path | Role |
| --- | --- |
| `python/phynetpy/` | The public package. Each ported module becomes a *facade*: same module name, same public and test-visible names, docstrings kept, implementation imported from `phynetpy._core`. |
| `src/lib.rs` | Module root. Every submodule is registered here already through a `register(m)` function. **Do not edit `lib.rs`, `src/kernels/mod.rs`, `src/graph/mod.rs` or `Cargo.toml`.** If you need a new crate dependency, say so in your report instead. |
| `src/graph/` | `Node`, `Edge`, `Network` (`network.rs`), graph algorithms (`algorithms.rs`), Newick (`newick.rs`), stub `utils.rs` for GraphUtils. |
| `src/kernels/` | `common.rs` (LOG_FLOOR, logaddexp, logsumexp, gij), `mask.rs` (lineage bitmasks), `msnc.rs` (coalescent DPs, a finished example), stubs `mpl.rs`, `seq.rs`, `snp.rs`. |
| `src/util.rs` | `warn(py, msg)`, `network_error(py, msg)` and friends (they raise the Python-defined exception classes), `py_float_repr`, `deepcopy`, `type_name`, `coerce_unit`. |
| `tests/` | The upstream suite, copied verbatim. It is the contract: `python -m pytest -m "not slow"` must stay at 1349 passed / 24 skipped. |

## Build and test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv"      # maturin installs into this venv
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv   # once
.venv/bin/python -m pip install -q maturin pytest numpy scipy scikit-learn matplotlib networkx biopython python-nexus newick pulp
.venv/bin/maturin develop            # compile + install (dev profile; add --release to benchmark)
.venv/bin/python -m pytest tests/test_x.py -q -p no:cacheprovider
.venv/bin/python -m pytest -m "not slow" -q -p no:cacheprovider      # the full contract, ~1 min
```

A bare `cargo build` without `VIRTUAL_ENV` fails at link time (it finds the
x86 Anaconda Python); `cargo check` is fine for type errors. `cargo test`
runs the pure-Rust unit tests.

## Conventions

- A port is complete when the facade keeps every name the tests or other
  modules import (grep `tests/` and `python/phynetpy/` for `from .<module>
  import` and `<module>.<name>`), the module's tests pass, and the full
  non-slow suite still passes.
- Numerical results must match the Python reference to floating-point
  tolerance; the tests encode the tolerances. Keep summation orders where a
  test compares against the Python path.
- Keep pure-Python *reference* implementations when a test monkeypatches or
  compares against them (see `_msnc_density.py`, which keeps
  `_apply_branch_coalescent_int_py`).
- Python classes that user code or other modules subclass must be
  `#[pyclass(subclass)]`. A Python subclass cannot call
  `super().__init__(args)` on a PyO3 class: give the Rust class a lenient
  `#[new]` (`*args, **kwargs`) and set the fields from the subclass instead
  (see `MSCBranchKernel` and `_GTLikelihoodEngine`).
- Exceptions: raise the Python-defined classes through `crate::util`
  (`network_error`, `node_error`, `edge_error`, `newick_error`) or standard
  ones (`PyValueError::new_err`). Warnings: `crate::util::warn(py, msg)?`.
- Float formatting that users see must match Python (`py_float_repr`).
- Never let Rust panic on a Python-reachable path. The search drivers catch
  `Exception` from moves and scorers and treat the proposal as invalid; a
  panic surfaces as `PanicException` (a `BaseException`) and aborts the run.
  Index maps with `.get()` and return a `PyErr` (`network_error`,
  `PyValueError`) where the reference raised, and tolerate stale ids where
  the reference read a stale `Node`/`Edge` object.
- Commit on your branch when done, with a message that says what moved to
  Rust, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## PyO3 0.30 notes (differences from older docs)

- Downcasts are `obj.cast::<T>()?` / `obj.cast_into::<T>()?` (not `downcast`).
- `PyList::new`, `PyTuple::new`, `PySet::new` return `PyResult<Bound<..>>`.
- `PyErr::warn(py, category.as_any(), &CString, stacklevel)`; use `util::warn`.
- `__hash__` may return `u64`; `__richcmp__(&self, other: &Bound<PyAny>, op: CompareOp, py)` with `pyo3::pyclass::CompareOp`.
- Methods that need the Python object of `self` take `slf: &Bound<'_, Self>` and borrow inside.
- numpy interop: crate `numpy` 0.30 (`PyReadonlyArray1<f64>`, `PyArray1::from_vec`, `to_pyarray`).
- Random numbers: callers pass `numpy.random.Generator` objects. Either call
  back into it (`rng.call_method1("integers", (0, n))`) when the random
  stream must stay shared with Python, or use `rand`/`rand_chacha` when the
  draws are internal to the kernel and a test only checks statistics.

## Network API from Rust (`crate::graph::{Network, Node, Edge}`)

`Network` is `#[pyclass(subclass)]`; the Python class `phynetpy.Network.Network`
subclasses it and is what `Network::new_object(py, unit)` instantiates. Take
`&Bound<'_, Network>` in functions and `borrow()` it.

- Nodes and edges carry process-unique ids (`uid`). `net.nodes: IndexMap<u64, NodeEntry>`
  (`entry.obj: Py<Node>`, `entry.in_edges` / `out_edges: IndexSet<u64>` of edge ids);
  `net.edges: IndexMap<u64, EdgeEntry>` (`entry.obj: Py<Edge>`, `entry.su`, `entry.du`).
  Iteration order of these maps is the order `V()` / `E()` return.
- Lookups: `uid_of_any(&Bound<PyAny>) -> Option<u64>`, `uid_of_bound(&Bound<Node>)`,
  `node_obj(uid) -> &Py<Node>`, `edge_obj(euid)`, `label(py, uid) -> String`,
  `in_deg(uid)`, `out_deg(uid)`, `children(uid) -> Vec<u64>`, `parents(uid)`,
  `in_edge_uids(uid)`, `out_edge_uids(uid)`, `root_uids()`, `leaf_uids()`,
  `root_uid(py) -> PyResult<u64>` (warns on several roots), `edge_between(py, su, du, gamma, tag)`,
  `first_edge_between(py, su, du) -> PyResult<u64>`.
- Algorithms: `topological_order_uids(py)`, `subtree_uids(uid)`, `leaf_descendants_uid(uid)`,
  `bfs_order(start, dfs) -> (order, dist)`, `mrca_uids(py, &[uid])`, `is_acyclic_impl()`,
  `newick_string(py, annotate)`, `edges_downstream(uid)`, `edges_upstream(uid)`.
- Mutation (needs the Python object for ownership): `net.add_node(slf.as_any(), &Bound<Node>)`,
  `net.add_edge(slf.as_any(), &Bound<Edge>)`, `net.remove_edge_uid(py, euid)`,
  `net.remove_node_uid(py, uid)`, `net.rename_uid(py, uid, name)`.
  `Network::copy_impl(&Bound<Network>) -> (Bound<Network>, Vec<(old, new)>)`.
- `Node` fields: `name`, `is_retic`, `t: Option<f64>`, `attrs: Py<PyDict>`, `seq`, `uid`.
  Build with `Node::make(py, name, is_retic, None, None, None)`; set time with `Node::set_time(&bound, t)?`.
- `Edge` fields: `src`, `dest: Py<Node>`, `length`, `gamma`, `weight: Option<f64>`, `tag: Option<String>`, `uid`.
  Build with `Edge::make(py, src, dest, Some(len), Some(gamma), Some(0.0), Some("no tag assigned yet".into()))`;
  `Edge::set_gamma(&bound, g)?`, `Edge::set_length(&bound, Some(l), false, false)?`.

Reading a Python-side `Network` method result is also fine from Rust
(`obj.call_method0("newick")`), but prefer the uid API in hot paths.
