// src_new/lib.rs

use pyo3::pymodule;

mod graph;
mod storage;
mod utils;

/// Main Python module entry point
#[pymodule]
pub fn _core(py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    // Core types
    m.add_class::<crate::graph::types::NodeId>()?;
    m.add_class::<crate::graph::types::EdgeId>()?;
    m.add_class::<crate::graph::types::GraphInfo>()?;
    
    // Collections and proxies
    m.add_class::<crate::graph::nodes::collection::NodeCollection>()?;
    m.add_class::<crate::graph::edges::collection::EdgeCollection>()?;
    m.add_class::<crate::graph::nodes::proxy::NodeProxy>()?;
    m.add_class::<crate::graph::edges::proxy::EdgeProxy>()?;
    
    // Managers
    m.add_class::<crate::graph::managers::attributes::AttributeManager>()?;
    m.add_class::<crate::graph::managers::filters::FilterManager>()?;
    
    // Main graph
    m.add_class::<crate::graph::core::FastGraph>()?;
    
    Ok(())
}
