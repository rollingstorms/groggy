//! Groggy Python Bindings
//! 
//! Fast graph library with statistical operations and memory-efficient processing.

use pyo3::prelude::*;

// Import all FFI modules
mod ffi;
mod module;

// Re-export main types
pub use ffi::api::graph::PyGraph;
pub use ffi::core::subgraph::PySubgraph;
pub use ffi::core::array::{PyGraphArray, PyGraphMatrix};
pub use ffi::core::accessors::{PyNodesAccessor, PyEdgesAccessor};
pub use ffi::core::views::{PyNodeView, PyEdgeView};
pub use ffi::types::{PyAttrValue, PyResultHandle, PyAttributeCollection};
pub use ffi::core::query::{PyAttributeFilter, PyNodeFilter, PyEdgeFilter};
pub use ffi::core::history::{PyCommit, PyBranchInfo, PyHistoryStatistics};
pub use ffi::api::graph_version::{PyHistoricalView};
pub use ffi::core::traversal::{PyTraversalResult, PyAggregationResult, PyGroupedAggregationResult};

/// A Python module implemented in Rust.
#[pymodule]
fn _groggy(py: Python, m: &PyModule) -> PyResult<()> {
    // Register core graph components
    m.add_class::<PyGraph>()?;
    m.add_class::<PySubgraph>()?;
    
    // Register array and matrix types
    m.add_class::<PyGraphArray>()?;
    m.add_class::<PyGraphMatrix>()?;
    
    // Register accessor and view types
    m.add_class::<PyNodesAccessor>()?;
    m.add_class::<PyEdgesAccessor>()?;
    m.add_class::<PyNodeView>()?;
    m.add_class::<PyEdgeView>()?;
    
    // Register type system
    m.add_class::<PyAttrValue>()?;
    m.add_class::<PyResultHandle>()?;
    m.add_class::<PyAttributeCollection>()?;
    
    // Register query and filter system
    m.add_class::<PyAttributeFilter>()?;
    m.add_class::<PyNodeFilter>()?;
    m.add_class::<PyEdgeFilter>()?;
    
    // Register version control system
    m.add_class::<PyCommit>()?;
    m.add_class::<PyBranchInfo>()?;
    m.add_class::<PyHistoryStatistics>()?;
    m.add_class::<PyHistoricalView>()?;
    
    // Register traversal and aggregation results
    m.add_class::<PyTraversalResult>()?;
    m.add_class::<PyAggregationResult>()?;
    m.add_class::<PyGroupedAggregationResult>()?;
    
    // Add aliases for Python imports - these are already added with correct names
    
    // Use the module registration function
    module::register_classes(py, m)?;
    
    Ok(())
}