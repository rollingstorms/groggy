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

// =============================================================================
// PyGraphTable - Python wrapper for GraphTable (composite)
// =============================================================================

/// Python wrapper for GraphTable
#[pyclass(name = "GraphTable", module = "groggy")]
#[derive(Clone)]
pub struct PyGraphTable {
    pub(crate) table: groggy::storage::table::GraphTable,
}

#[pymethods]
impl PyGraphTable {
    /// Get number of total rows (nodes + edges)
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns (max of nodes and edges)
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Get NodesTable component
    pub fn nodes(&self) -> PyNodesTable {
        PyNodesTable { table: self.table.nodes().clone() }
    }
    
    /// Get EdgesTable component  
    pub fn edges(&self) -> PyEdgesTable {
        PyEdgesTable { table: self.table.edges().clone() }
    }
    
    /// Validate the GraphTable and return report
    pub fn validate(&self) -> PyResult<String> {
        let report = self.table.validate();
        Ok(format!("{:?}", report))
    }
    
    /// Convert back to Graph
    pub fn to_graph(&self) -> PyResult<crate::ffi::api::graph::PyGraph> {
        let graph = self.table.clone().to_graph()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(crate::ffi::api::graph::PyGraph {
            inner: std::rc::Rc::new(std::cell::RefCell::new(graph)),
            cached_view: std::cell::RefCell::new(None),
        })
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "GraphTable[{} nodes, {} edges]", 
            self.table.nodes().nrows(),
            self.table.edges().nrows()
        )
    }
    
    /// Create PyGraphTable from GraphTable (used by accessors)
    pub fn from_graph_table(table: groggy::storage::table::GraphTable) -> Self {
        Self { table }
    }
    
    /// Merge multiple GraphTables into one
    #[staticmethod]
    pub fn merge(tables: Vec<PyGraphTable>) -> PyResult<PyGraphTable> {
        let rust_tables: Vec<groggy::storage::table::GraphTable> = 
            tables.into_iter().map(|t| t.table).collect();
            
        let merged = groggy::storage::table::GraphTable::merge(rust_tables)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table: merged })
    }
    
    /// Merge with conflict resolution strategy
    #[staticmethod] 
    pub fn merge_with_strategy(tables: Vec<PyGraphTable>, strategy: &str) -> PyResult<PyGraphTable> {
        use groggy::storage::table::ConflictResolution;
        
        let conflict_strategy = match strategy.to_lowercase().as_str() {
            "fail" => ConflictResolution::Fail,
            "keep_first" => ConflictResolution::KeepFirst,
            "keep_second" => ConflictResolution::KeepSecond,
            "merge_attributes" => ConflictResolution::MergeAttributes,
            "domain_prefix" => ConflictResolution::DomainPrefix,
            "auto_remap" => ConflictResolution::AutoRemap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown conflict resolution strategy: {}", strategy)
            ))
        };
        
        let rust_tables: Vec<groggy::storage::table::GraphTable> = 
            tables.into_iter().map(|t| t.table).collect();
            
        let merged = groggy::storage::table::GraphTable::merge_with_strategy(rust_tables, conflict_strategy)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table: merged })
    }
    
    /// Merge with another GraphTable
    pub fn merge_with(&mut self, other: PyGraphTable, strategy: &str) -> PyResult<()> {
        use groggy::storage::table::ConflictResolution;
        
        let conflict_strategy = match strategy.to_lowercase().as_str() {
            "fail" => ConflictResolution::Fail,
            "keep_first" => ConflictResolution::KeepFirst, 
            "keep_second" => ConflictResolution::KeepSecond,
            "merge_attributes" => ConflictResolution::MergeAttributes,
            "domain_prefix" => ConflictResolution::DomainPrefix,
            "auto_remap" => ConflictResolution::AutoRemap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown conflict resolution strategy: {}", strategy)
            ))
        };
        
        self.table.merge_with(other.table, conflict_strategy)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(())
    }
    
    /// Create federated GraphTable from multiple bundle paths
    #[staticmethod]
    pub fn from_federated_bundles(bundle_paths: Vec<&str>, domain_names: Option<Vec<String>>) -> PyResult<PyGraphTable> {
        use std::path::Path;
        
        let paths: Vec<&Path> = bundle_paths.iter().map(|s| Path::new(s)).collect();
        
        let table = groggy::storage::table::GraphTable::from_federated_bundles(paths, domain_names)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table })
    }
    
    /// Save GraphTable as bundle to disk
    pub fn save_bundle(&self, path: &str) -> PyResult<()> {
        use std::path::Path;
        
        self.table.save_bundle(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(())
    }
    
    /// Load GraphTable from bundle on disk
    #[staticmethod]
    pub fn load_bundle(path: &str) -> PyResult<PyGraphTable> {
        use std::path::Path;
        
        let table = groggy::storage::table::GraphTable::load_bundle(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table })
    }
    
    /// Get graph statistics  
    pub fn stats(&self) -> std::collections::HashMap<String, usize> {
        self.table.stats()
    }
}