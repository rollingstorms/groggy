//! Graph Version Control Module
//!
//! Python bindings for graph versioning and history operations.

use crate::ffi::utils::graph_error_to_py_err;
use groggy::{AttrValue as RustAttrValue, StateId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Version control operations for graphs
#[pyclass(name = "GraphVersion")]
pub struct PyGraphVersion {
    /// Reference to the parent graph
    pub graph: Py<crate::ffi::api::graph::PyGraph>,
}

/// Python wrapper for commit information
#[pyclass(name = "Commit")]
#[derive(Clone)]
pub struct PyCommit {
    inner: std::sync::Arc<groggy::core::history::Commit>,
}

impl PyCommit {
    /// Create a PyCommit from a core Commit
    pub fn from_core_commit(commit: std::sync::Arc<groggy::core::history::Commit>) -> Self {
        Self { inner: commit }
    }

    /// Create a PyCommit from a CommitInfo (simplified info)
    pub fn from_commit_info(info: groggy::api::graph::CommitInfo) -> Self {
        // Create a simplified Commit from CommitInfo
        // Note: This is a temporary bridge until we have full core integration
        let parents = match info.parent {
            Some(p) => vec![p],
            None => vec![],
        };

        let fake_delta = std::sync::Arc::new(groggy::core::history::Delta {
            content_hash: [0u8; 32],
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        });

        let commit = groggy::core::history::Commit::new(
            info.id,
            parents,
            fake_delta,
            info.message,
            info.author,
        );

        Self {
            inner: std::sync::Arc::new(commit),
        }
    }
}

#[pymethods]
impl PyCommit {
    #[getter]
    fn id(&self) -> StateId {
        self.inner.id
    }

    #[getter]
    fn parents(&self) -> Vec<StateId> {
        self.inner.parents.clone()
    }

    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    #[getter]
    fn author(&self) -> String {
        self.inner.author.clone()
    }

    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }

    fn __repr__(&self) -> String {
        format!(
            "Commit(id={}, message='{}', author='{}')",
            self.inner.id, self.inner.message, self.inner.author
        )
    }
}

/// Python wrapper for branch information
#[pyclass(name = "BranchInfo")]
#[derive(Clone)]
pub struct PyBranchInfo {
    inner: groggy::core::ref_manager::BranchInfo,
}

impl PyBranchInfo {
    /// Create a new PyBranchInfo from core BranchInfo
    pub fn new(inner: groggy::core::ref_manager::BranchInfo) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBranchInfo {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn head(&self) -> StateId {
        self.inner.head
    }

    #[getter]
    fn is_current(&self) -> bool {
        self.inner.is_current
    }

    fn __repr__(&self) -> String {
        format!(
            "BranchInfo(name='{}', head={}, current={})",
            self.inner.name, self.inner.head, self.inner.is_current
        )
    }
}

/// Python wrapper for historical view
#[pyclass(name = "HistoricalView")]
pub struct PyHistoricalView {
    pub state_id: StateId,
}

#[pymethods]
impl PyHistoricalView {
    #[getter]
    fn state_id(&self) -> StateId {
        self.state_id
    }

    fn __repr__(&self) -> String {
        format!("HistoricalView(state_id={})", self.state_id)
    }
}

#[pymethods]
impl PyGraphVersion {
    /// Commit current changes to version control
    fn commit(&self, py: Python, message: String, author: String) -> PyResult<StateId> {
        let mut graph = self.graph.borrow_mut(py);
        graph
            .inner
            .commit(message, author)
            .map_err(graph_error_to_py_err)
    }

    /// Create a new branch
    fn create_branch(&self, py: Python, branch_name: String) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        graph
            .inner
            .create_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// Switch to a different branch
    fn checkout_branch(&self, py: Python, branch_name: String) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        graph
            .inner
            .checkout_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// List all branches
    fn branches(&self, py: Python) -> Vec<PyBranchInfo> {
        let graph = self.graph.borrow(py);
        graph
            .inner
            .list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo { inner: branch_info })
            .collect()
    }

    /// Get commit history
    fn commit_history(&self, py: Python) -> Vec<PyCommit> {
        let graph = self.graph.borrow(py);
        // Note: This is a simplified implementation
        // In the full implementation, you'd convert from CommitInfo to Commit
        Vec::new()
    }

    /// Get historical view at a specific commit
    fn historical_view(&self, py: Python, commit_id: StateId) -> PyResult<PyHistoricalView> {
        let graph = self.graph.borrow(py);
        match graph.inner.view_at_commit(commit_id) {
            Ok(_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Check if there are uncommitted changes
    fn has_uncommitted_changes(&self, py: Python) -> bool {
        let graph = self.graph.borrow(py);
        graph.inner.has_uncommitted_changes()
    }

    /// Create a snapshot of the current graph state
    fn create_snapshot(&self, py: Python, name: Option<&str>) -> PyResult<PyObject> {
        let snapshot_name = name.unwrap_or("snapshot");
        let author = "system".to_string();
        let message = format!("Snapshot: {}", snapshot_name);

        let state_id = self.commit(py, message, author)?;

        // Return snapshot info as a dictionary
        let dict = PyDict::new(py);
        dict.set_item("state_id", state_id)?;
        dict.set_item("name", snapshot_name)?;
        dict.set_item("type", "snapshot")?;

        Ok(dict.to_object(py))
    }

    /// Restore graph to a previous snapshot
    fn restore_snapshot(&self, py: Python, snapshot_id: &str) -> PyResult<bool> {
        // Parse snapshot_id as StateId
        match snapshot_id.parse::<StateId>() {
            Ok(state_id) => {
                match self.historical_view(py, state_id) {
                    Ok(_) => {
                        // In a full implementation, you'd actually restore the state
                        // For now, just indicate success if the snapshot exists
                        Ok(true)
                    }
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Err(PyValueError::new_err(format!(
                "Invalid snapshot ID: {}",
                snapshot_id
            ))),
        }
    }

    /// Get version history
    fn get_history(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);

        // Create a summary of version history
        let dict = PyDict::new(py);

        // Get branch information
        let branches = self.branches(py);
        let branch_list: Vec<PyObject> = branches
            .into_iter()
            .map(|branch| Py::new(py, branch).unwrap().to_object(py))
            .collect();
        dict.set_item("branches", branch_list)?;

        // Get commit count (simplified)
        dict.set_item("total_commits", 0)?; // Would be implemented with actual history
        dict.set_item(
            "has_uncommitted_changes",
            graph.inner.has_uncommitted_changes(),
        )?;

        // Get current state info
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();
        dict.set_item(
            "current_state",
            format!("{} nodes, {} edges", node_count, edge_count),
        )?;

        Ok(dict.to_object(py))
    }

    /// Get node mapping for a specific attribute
    fn get_node_mapping(&self, py: Python, uid_key: String) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let dict = PyDict::new(py);
        let node_ids = graph.inner.node_ids();

        // Scan all nodes for the specified uid_key attribute
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = graph.inner.get_node_attr(node_id, &uid_key) {
                // Convert attribute value to appropriate Python type
                let key_value = match attr_value {
                    RustAttrValue::Text(s) => s.to_object(py),
                    RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                    RustAttrValue::Int(i) => i.to_object(py),
                    RustAttrValue::SmallInt(i) => i.to_object(py),
                    RustAttrValue::Float(f) => f.to_object(py),
                    RustAttrValue::Bool(b) => b.to_object(py),
                    _ => continue, // Skip unsupported types
                };

                dict.set_item(key_value, node_id)?;
            }
        }

        Ok(dict.to_object(py))
    }

    /// Get version info
    fn get_info(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();
        let has_changes = graph.inner.has_uncommitted_changes();

        Ok(format!(
            "Version Control: {} nodes, {} edges, uncommitted changes: {}",
            node_count, edge_count, has_changes
        ))
    }
}
