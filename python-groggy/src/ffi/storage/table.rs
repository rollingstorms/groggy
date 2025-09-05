//! Python FFI for BaseTable system

use pyo3::prelude::*;
use crate::ffi::storage::array::PyBaseArray;
use groggy::storage::table::{BaseTable, NodesTable, EdgesTable, Table, TableIterator};
use groggy::types::{NodeId, EdgeId};

// =============================================================================
// PyBaseTable - Python wrapper for BaseTable
// =============================================================================

/// Python wrapper for BaseTable
#[pyclass(name = "BaseTable", module = "groggy")]
#[derive(Clone)]
pub struct PyBaseTable {
    pub(crate) table: BaseTable,
}

#[pymethods]
impl PyBaseTable {
    /// Create a new empty BaseTable
    #[new]
    pub fn new() -> Self {
        Self {
            table: BaseTable::new(),
        }
    }
    
    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.table.column_names().to_vec()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Check if column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.table.has_column(name)
    }
    
    /// Get first n rows
    pub fn head(&self, n: usize) -> Self {
        Self {
            table: self.table.head(n),
        }
    }
    
    /// Get last n rows
    pub fn tail(&self, n: usize) -> Self {
        Self {
            table: self.table.tail(n),
        }
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyBaseTableIterator {
        PyBaseTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("BaseTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
}

// =============================================================================
// PyBaseTableIterator - Python wrapper for TableIterator<BaseTable>
// =============================================================================

/// Python wrapper for BaseTable iterator
#[pyclass(name = "BaseTableIterator", module = "groggy")]
pub struct PyBaseTableIterator {
    pub(crate) iterator: TableIterator<BaseTable>,
}

#[pymethods]
impl PyBaseTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyBaseTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyBaseTable { table: result })
    }
}

// =============================================================================
// PyNodesTable - Python wrapper for NodesTable
// =============================================================================

/// Python wrapper for NodesTable
#[pyclass(name = "NodesTable", module = "groggy")]
#[derive(Clone)]
pub struct PyNodesTable {
    pub(crate) table: NodesTable,
}

#[pymethods]
impl PyNodesTable {
    /// Create new NodesTable from node IDs
    #[new]
    pub fn new(node_ids: Vec<NodeId>) -> Self {
        Self {
            table: NodesTable::new(node_ids),
        }
    }
    
    /// Get node IDs
    pub fn node_ids(&self) -> PyResult<Vec<NodeId>> {
        self.table.node_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyNodesTableIterator {
        PyNodesTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NodesTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
}

// =============================================================================
// PyNodesTableIterator - Python wrapper for TableIterator<NodesTable>
// =============================================================================

/// Python wrapper for NodesTable iterator
#[pyclass(name = "NodesTableIterator", module = "groggy")]
pub struct PyNodesTableIterator {
    pub(crate) iterator: TableIterator<NodesTable>,
}

#[pymethods]
impl PyNodesTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyNodesTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyNodesTable { table: result })
    }
}

// =============================================================================
// PyEdgesTable - Python wrapper for EdgesTable
// =============================================================================

/// Python wrapper for EdgesTable
#[pyclass(name = "EdgesTable", module = "groggy")]
#[derive(Clone)]
pub struct PyEdgesTable {
    pub(crate) table: EdgesTable,
}

#[pymethods]
impl PyEdgesTable {
    /// Create new EdgesTable from edge tuples
    #[new]
    pub fn new(edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        Self {
            table: EdgesTable::new(edges),
        }
    }
    
    /// Get edge IDs
    pub fn edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        self.table.edge_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyEdgesTableIterator {
        PyEdgesTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("EdgesTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
}

// =============================================================================
// PyEdgesTableIterator - Python wrapper for TableIterator<EdgesTable>
// =============================================================================

/// Python wrapper for EdgesTable iterator
#[pyclass(name = "EdgesTableIterator", module = "groggy")]
pub struct PyEdgesTableIterator {
    pub(crate) iterator: TableIterator<EdgesTable>,
}

#[pymethods]
impl PyEdgesTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyEdgesTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyEdgesTable { table: result })
    }
}