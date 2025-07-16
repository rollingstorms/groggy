// src_new/lib.rs

use pyo3::pymodule;

mod graph;
mod storage;
mod utils;

/// Main Python module entry point
#[pymodule]
pub fn _core(py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    m.add_class::<crate::graph::nodes::collection::NodeCollection>()?;
    m.add_class::<crate::graph::edges::collection::EdgeCollection>()?;
    m.add_class::<crate::graph::nodes::proxy::NodeProxy>()?;
    m.add_class::<crate::graph::edges::proxy::EdgeProxy>()?;
    m.add_class::<crate::graph::managers::attributes::AttributeManager>()?;
    m.add_class::<crate::graph::managers::filters::FilterManager>()?;
    m.add_class::<crate::graph::core::FastGraph>()?;
    // Add any other core classes or submodules here as needed
    Ok(())
}
