//! History FFI Bindings
//!
//! Python bindings for version control and history operations.

use groggy::core::history::{Commit, HistoryStatistics};
use groggy::core::ref_manager::BranchInfo;
use groggy::{EdgeId, NodeId, StateId};
use pyo3::prelude::*;

/// Python wrapper for Commit
#[pyclass(name = "Commit")]
#[derive(Clone)]
pub struct PyCommit {
    pub inner: std::sync::Arc<Commit>,
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

    fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    fn is_merge(&self) -> bool {
        self.inner.is_merge()
    }

    fn __repr__(&self) -> String {
        format!(
            "Commit(id={}, message='{}', author='{}')",
            self.inner.id, self.inner.message, self.inner.author
        )
    }
}

/// Python wrapper for BranchInfo
#[pyclass(name = "BranchInfo")]
#[derive(Clone)]
pub struct PyBranchInfo {
    pub inner: BranchInfo,
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
    fn is_default(&self) -> bool {
        self.inner.is_default
    }

    #[getter]
    fn is_current(&self) -> bool {
        self.inner.is_current
    }

    fn __repr__(&self) -> String {
        format!(
            "BranchInfo(name='{}', head={})",
            self.inner.name, self.inner.head
        )
    }
}

/// Python wrapper for HistoryStatistics
#[pyclass(name = "HistoryStatistics")]
#[derive(Clone)]
pub struct PyHistoryStatistics {
    pub inner: HistoryStatistics,
}

#[pymethods]
impl PyHistoryStatistics {
    #[getter]
    fn total_commits(&self) -> usize {
        self.inner.total_commits
    }

    #[getter]
    fn total_branches(&self) -> usize {
        self.inner.total_branches
    }

    #[getter]
    fn total_tags(&self) -> usize {
        self.inner.total_tags
    }

    #[getter]
    fn storage_efficiency(&self) -> f64 {
        self.inner.storage_efficiency
    }

    #[getter]
    fn oldest_commit_age(&self) -> u64 {
        self.inner.oldest_commit_age
    }

    #[getter]
    fn newest_commit_age(&self) -> u64 {
        self.inner.newest_commit_age
    }

    fn __repr__(&self) -> String {
        format!(
            "HistoryStatistics(commits={}, branches={}, efficiency={:.2})",
            self.inner.total_commits, self.inner.total_branches, self.inner.storage_efficiency
        )
    }
}

/// Python wrapper for HistoricalView
#[pyclass(name = "HistoricalView")]
pub struct PyHistoricalView {
    // Store the state ID that this view represents
    pub state_id: StateId,
    // For actual graph operations, we'll need to call back to the graph
    // In a full implementation, this would contain a HistoricalView<'graph>
}

#[pymethods]
impl PyHistoricalView {
    #[getter]
    fn state_id(&self) -> StateId {
        self.state_id
    }

    /// Get nodes from this historical state
    /// Note: This is a simplified implementation. In practice, you'd need
    /// access to the graph to reconstruct the state.
    fn get_node_ids(&self) -> PyResult<Vec<NodeId>> {
        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
    }

    /// Get edges from this historical state
    fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
    }

    fn __repr__(&self) -> String {
        format!("HistoricalView(state_id={})", self.state_id)
    }
}
