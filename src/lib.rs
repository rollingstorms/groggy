use pyo3::prelude::*;

mod graph;
mod storage;
mod utils;

use graph::FastGraph;
use storage::{ContentPool, GraphStore};

/// Groggy Core - High-performance graph operations implemented in Rust
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastGraph>()?;
    m.add_class::<ContentPool>()?;
    m.add_class::<GraphStore>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
