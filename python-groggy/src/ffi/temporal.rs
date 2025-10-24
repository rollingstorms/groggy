use crate::ffi::subgraphs::subgraph::PySubgraph;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::temporal::{ExistenceIndex, IndexStatistics, TemporalIndex, TemporalSnapshot};
use groggy::types::{AttrName, EdgeId, NodeId, StateId};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "TemporalSnapshot", unsendable)]
pub struct PyTemporalSnapshot {
    pub inner: TemporalSnapshot,
}

#[pymethods]
impl PyTemporalSnapshot {
    #[getter]
    fn commit_id(&self) -> u64 {
        self.inner.lineage().commit_id
    }

    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.lineage().timestamp
    }

    #[getter]
    fn author(&self) -> &str {
        &self.inner.lineage().author
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.lineage().message
    }

    fn parents(&self) -> Vec<u64> {
        self.inner.lineage().parent_commits.to_vec()
    }

    fn node_exists(&self, node_id: u64) -> bool {
        self.inner.existence().contains_node(node_id as NodeId)
    }

    fn edge_exists(&self, edge_id: u64) -> bool {
        self.inner.existence().contains_edge(edge_id as EdgeId)
    }

    fn node_attr(&self, node_id: u64, attr: &str) -> Option<PyAttrValue> {
        self.inner
            .node_attr(node_id as NodeId, &AttrName::from(attr.to_string()))
            .map(PyAttrValue::new)
    }

    fn edge_attr(&self, edge_id: u64, attr: &str) -> Option<PyAttrValue> {
        self.inner
            .edge_attr(edge_id as EdgeId, &AttrName::from(attr.to_string()))
            .map(PyAttrValue::new)
    }

    fn neighbors(&self, node_id: u64) -> Vec<u64> {
        self.inner
            .neighbors(node_id as NodeId)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n as u64)
            .collect()
    }

    fn neighbors_bulk(&self, nodes: Vec<u64>) -> HashMap<u64, Vec<u64>> {
        self.inner
            .neighbors_bulk(&nodes.iter().map(|id| *id as NodeId).collect::<Vec<_>>())
            .into_iter()
            .map(|(node, neigh)| (node as u64, neigh.into_iter().map(|n| n as u64).collect()))
            .collect()
    }

    fn existence(&self) -> PyExistenceIndex {
        PyExistenceIndex {
            inner: self.inner.existence().clone(),
        }
    }

    fn as_subgraph(&self, py: Python<'_>) -> PyResult<Py<PySubgraph>> {
        let subgraph = self.inner.as_subgraph().map_err(graph_error_to_py_err)?;
        let py_subgraph = PySubgraph::from_core_subgraph(subgraph)?;
        Py::new(py, py_subgraph)
    }
}

#[pyclass(name = "ExistenceIndex", unsendable)]
pub struct PyExistenceIndex {
    pub inner: ExistenceIndex,
}

#[pymethods]
impl PyExistenceIndex {
    fn contains_node(&self, node_id: u64) -> bool {
        self.inner.contains_node(node_id as NodeId)
    }

    fn contains_edge(&self, edge_id: u64) -> bool {
        self.inner.contains_edge(edge_id as EdgeId)
    }

    fn nodes(&self) -> Vec<u64> {
        self.inner.nodes().iter().map(|id| *id as u64).collect()
    }

    fn edges(&self) -> Vec<u64> {
        self.inner.edges().iter().map(|id| *id as u64).collect()
    }
}

/// Python wrapper for TemporalIndex
#[pyclass(name = "TemporalIndex", unsendable)]
pub struct PyTemporalIndex {
    pub inner: TemporalIndex,
}

#[pymethods]
impl PyTemporalIndex {
    fn node_exists_at(&self, node_id: u64, commit_id: u64) -> bool {
        self.inner
            .node_exists_at(node_id as NodeId, commit_id as StateId)
    }

    fn edge_exists_at(&self, edge_id: u64, commit_id: u64) -> bool {
        self.inner
            .edge_exists_at(edge_id as EdgeId, commit_id as StateId)
    }

    fn nodes_at_commit(&self, commit_id: u64) -> Vec<u64> {
        self.inner
            .nodes_at_commit(commit_id as StateId)
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn edges_at_commit(&self, commit_id: u64) -> Vec<u64> {
        self.inner
            .edges_at_commit(commit_id as StateId)
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn neighbors_at_commit(&self, node_id: u64, commit_id: u64) -> Vec<u64> {
        self.inner
            .neighbors_at_commit(node_id as NodeId, commit_id as StateId)
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn neighbors_bulk_at_commit(&self, nodes: Vec<u64>, commit_id: u64) -> HashMap<u64, Vec<u64>> {
        let node_ids: Vec<NodeId> = nodes.iter().map(|id| *id as NodeId).collect();
        self.inner
            .neighbors_bulk_at_commit(&node_ids, commit_id as StateId)
            .into_iter()
            .map(|(node, neighbors)| {
                (
                    node as u64,
                    neighbors.into_iter().map(|n| n as u64).collect(),
                )
            })
            .collect()
    }

    fn neighbors_in_window(&self, node_id: u64, start_commit: u64, end_commit: u64) -> Vec<u64> {
        self.inner
            .neighbors_in_window(
                node_id as NodeId,
                start_commit as StateId,
                end_commit as StateId,
            )
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn node_attr_at_commit(&self, node_id: u64, attr: &str, commit_id: u64) -> Option<PyAttrValue> {
        self.inner
            .node_attr_at_commit(
                node_id as NodeId,
                &AttrName::from(attr.to_string()),
                commit_id as StateId,
            )
            .map(PyAttrValue::new)
    }

    fn node_attr_history(
        &self,
        node_id: u64,
        attr: &str,
        from_commit: u64,
        to_commit: u64,
    ) -> Vec<(u64, PyAttrValue)> {
        self.inner
            .node_attr_history(
                node_id as NodeId,
                &AttrName::from(attr.to_string()),
                from_commit as StateId,
                to_commit as StateId,
            )
            .into_iter()
            .map(|(cid, val)| (cid, PyAttrValue::new(val)))
            .collect()
    }

    fn edge_attr_at_commit(&self, edge_id: u64, attr: &str, commit_id: u64) -> Option<PyAttrValue> {
        self.inner
            .edge_attr_at_commit(
                edge_id as EdgeId,
                &AttrName::from(attr.to_string()),
                commit_id as StateId,
            )
            .map(PyAttrValue::new)
    }

    fn edge_attr_history(
        &self,
        edge_id: u64,
        attr: &str,
        from_commit: u64,
        to_commit: u64,
    ) -> Vec<(u64, PyAttrValue)> {
        self.inner
            .edge_attr_history(
                edge_id as EdgeId,
                &AttrName::from(attr.to_string()),
                from_commit as StateId,
                to_commit as StateId,
            )
            .into_iter()
            .map(|(cid, val)| (cid, PyAttrValue::new(val)))
            .collect()
    }

    fn commits_in_time_range(&self, start_ts: u64, end_ts: u64) -> Vec<u64> {
        self.inner
            .commits_in_time_range(start_ts, end_ts)
            .into_iter()
            .collect()
    }

    fn nodes_changed_in_commit(&self, commit_id: u64) -> Vec<u64> {
        self.inner
            .nodes_changed_in_commit(commit_id as StateId)
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn edges_changed_in_commit(&self, commit_id: u64) -> Vec<u64> {
        self.inner
            .edges_changed_in_commit(commit_id as StateId)
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn statistics(&self) -> PyIndexStatistics {
        PyIndexStatistics {
            inner: self.inner.statistics(),
        }
    }
}

/// Python wrapper for IndexStatistics
#[pyclass(name = "IndexStatistics")]
#[derive(Clone)]
pub struct PyIndexStatistics {
    pub inner: IndexStatistics,
}

#[pymethods]
impl PyIndexStatistics {
    #[getter]
    fn total_nodes(&self) -> usize {
        self.inner.total_nodes
    }

    #[getter]
    fn total_edges(&self) -> usize {
        self.inner.total_edges
    }

    #[getter]
    fn total_commits(&self) -> usize {
        self.inner.total_commits
    }

    #[getter]
    fn node_attr_timelines(&self) -> usize {
        self.inner.node_attr_timelines
    }

    #[getter]
    fn edge_attr_timelines(&self) -> usize {
        self.inner.edge_attr_timelines
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexStatistics(nodes={}, edges={}, commits={}, node_attr_timelines={}, edge_attr_timelines={})",
            self.inner.total_nodes,
            self.inner.total_edges,
            self.inner.total_commits,
            self.inner.node_attr_timelines,
            self.inner.edge_attr_timelines
        )
    }
}

// === Temporal Context Types ===

use groggy::algorithms::{ChangedEntities, TemporalDelta, TemporalScope};

/// Python wrapper for TemporalScope
#[pyclass(name = "TemporalScope", unsendable)]
#[derive(Clone)]
pub struct PyTemporalScope {
    pub inner: TemporalScope,
}

#[pymethods]
impl PyTemporalScope {
    #[new]
    fn new(current_commit: u64, window: Option<(u64, u64)>) -> Self {
        let scope = if let Some((start, end)) = window {
            TemporalScope::with_window(current_commit, start, end)
        } else {
            TemporalScope::at_commit(current_commit)
        };
        Self { inner: scope }
    }

    #[getter]
    fn current_commit(&self) -> u64 {
        self.inner.current_commit
    }

    #[getter]
    fn window(&self) -> Option<(u64, u64)> {
        self.inner.window
    }

    fn has_window(&self) -> bool {
        self.inner.has_window()
    }

    fn has_reference(&self) -> bool {
        self.inner.has_reference()
    }

    fn window_size(&self) -> Option<usize> {
        self.inner.window_size()
    }

    fn with_metadata(mut slf: PyRefMut<Self>, key: String, value: String) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_metadata(key, value);
        slf
    }

    fn __repr__(&self) -> String {
        if let Some((start, end)) = self.inner.window {
            format!(
                "TemporalScope(commit={}, window=({}, {}))",
                self.inner.current_commit, start, end
            )
        } else {
            format!("TemporalScope(commit={})", self.inner.current_commit)
        }
    }
}

/// Python wrapper for TemporalDelta
#[pyclass(name = "TemporalDelta")]
#[derive(Clone)]
pub struct PyTemporalDelta {
    pub inner: TemporalDelta,
}

#[pymethods]
impl PyTemporalDelta {
    #[getter]
    #[allow(clippy::wrong_self_convention)] // Getter for field, not a constructor
    fn from_commit(&self) -> u64 {
        self.inner.from_commit
    }

    #[getter]
    fn to_commit(&self) -> u64 {
        self.inner.to_commit
    }

    #[getter]
    fn nodes_added(&self) -> Vec<u64> {
        self.inner.nodes_added.iter().map(|id| *id as u64).collect()
    }

    #[getter]
    fn nodes_removed(&self) -> Vec<u64> {
        self.inner
            .nodes_removed
            .iter()
            .map(|id| *id as u64)
            .collect()
    }

    #[getter]
    fn edges_added(&self) -> Vec<u64> {
        self.inner.edges_added.iter().map(|id| *id as u64).collect()
    }

    #[getter]
    fn edges_removed(&self) -> Vec<u64> {
        self.inner
            .edges_removed
            .iter()
            .map(|id| *id as u64)
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn summary(&self) -> String {
        self.inner.summary()
    }

    fn affected_nodes(&self) -> Vec<u64> {
        self.inner
            .affected_nodes()
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn affected_edges(&self) -> Vec<u64> {
        self.inner
            .affected_edges()
            .into_iter()
            .map(|id| id as u64)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TemporalDelta(commits={}→{}, {})",
            self.inner.from_commit,
            self.inner.to_commit,
            self.inner.summary()
        )
    }
}

/// Python wrapper for ChangedEntities
#[pyclass(name = "ChangedEntities")]
#[derive(Clone)]
pub struct PyChangedEntities {
    pub inner: ChangedEntities,
}

#[pymethods]
impl PyChangedEntities {
    #[getter]
    fn modified_nodes(&self) -> Vec<u64> {
        self.inner
            .modified_nodes
            .iter()
            .map(|id| *id as u64)
            .collect()
    }

    #[getter]
    fn modified_edges(&self) -> Vec<u64> {
        self.inner
            .modified_edges
            .iter()
            .map(|id| *id as u64)
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn total_changes(&self) -> usize {
        self.inner.total_changes()
    }

    fn node_change_type(&self, node_id: u64) -> Option<String> {
        self.inner
            .node_change_types
            .get(&(node_id as NodeId))
            .map(|ct| format!("{:?}", ct))
    }

    fn edge_change_type(&self, edge_id: u64) -> Option<String> {
        self.inner
            .edge_change_types
            .get(&(edge_id as EdgeId))
            .map(|ct| format!("{:?}", ct))
    }

    fn __repr__(&self) -> String {
        format!(
            "ChangedEntities(nodes={}, edges={})",
            self.inner.modified_nodes.len(),
            self.inner.modified_edges.len()
        )
    }
}
