//! PhyNetPy's Rust core (``phynetpy._core``).
//!
//! The Python package ``phynetpy`` is the public API; this extension module
//! holds the graph structures, the Newick parser, and (in later modules)
//! the likelihood kernels and search engines.

use pyo3::prelude::*;

pub mod dimple;
pub mod graph;
pub mod io;
pub mod metrics;
pub mod kernels;
pub mod search;
pub mod sim;
pub mod util;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "PhyNetPy Rust core: graph structures and numerical kernels.")?;
    m.add_class::<graph::Node>()?;
    m.add_class::<graph::Edge>()?;
    m.add_class::<graph::Branch>()?;
    m.add_class::<graph::NodeSet>()?;
    m.add_class::<graph::EdgeSet>()?;
    m.add_class::<graph::Network>()?;
    m.add_function(wrap_pyfunction!(util::stable_hash_py, m)?)?;
    m.add_function(wrap_pyfunction!(util::_register_exceptions, m)?)?;
    m.add_function(wrap_pyfunction!(util::_register_newick_error, m)?)?;
    m.add_function(wrap_pyfunction!(util::_register_network_class, m)?)?;
    m.add_function(wrap_pyfunction!(graph::newick::parse_newick, m)?)?;
    m.add_function(wrap_pyfunction!(graph::newick::complete_hybrid_gammas_py, m)?)?;
    kernels::msnc::register(m)?;
    kernels::mpl::register(m)?;
    kernels::seq::register(m)?;
    kernels::snp::register(m)?;
    graph::utils::register(m)?;
    kernels::allop::register(m)?;
    kernels::chain::register(m)?;
    kernels::optimize::register(m)?;
    io::register(m)?;
    search::moves::register(m)?;
    sim::birthdeath::register(m)?;
    sim::seq::register(m)?;
    sim::snp::register(m)?;
    graph::genetrees::register(m)?;
    metrics::register(m)?;
    dimple::register(m)?;
    search::mcmc_seq::register(m)?;
    search::mcmc_gt::register(m)?;
    Ok(())
}
