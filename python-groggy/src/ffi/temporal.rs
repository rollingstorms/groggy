use crate::ffi::subgraphs::subgraph::PySubgraph;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::temporal::{ExistenceIndex, TemporalSnapshot};
use groggy::types::{AttrName, EdgeId, NodeId};
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
        self.inner.lineage().commit_id as u64
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
        self.inner
            .lineage()
            .parent_commits
            .iter()
            .map(|id| *id as u64)
            .collect()
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
        let subgraph = self
            .inner
            .as_subgraph()
            .map_err(graph_error_to_py_err)?;
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
