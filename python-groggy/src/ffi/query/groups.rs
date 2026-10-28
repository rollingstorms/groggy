//! Strict controls and detached native calls for immutable group membership.
use super::query::nonnegative_count;
use crate::ffi::{subgraphs::view::PyGraphView, utils::graph_error_to_py_err};
use groggy::query::GroupMembershipIndex;
use pyo3::prelude::*;

#[pyclass(name = "GroupMembershipIndex")]
#[derive(Clone)]
pub struct PyGroupMembershipIndex {
    pub inner: GroupMembershipIndex,
}
#[pymethods]
impl PyGroupMembershipIndex {
    #[getter]
    fn view(&self) -> PyGraphView {
        PyGraphView {
            inner: self.inner.view().clone(),
        }
    }
    #[getter]
    fn group_ids(&self, py: Python<'_>) -> Vec<usize> {
        py.allow_threads(|| self.inner.group_ids())
    }
    #[getter]
    fn root_groups(&self, py: Python<'_>) -> Vec<usize> {
        py.allow_threads(|| self.inner.root_groups())
    }
    fn group_view(&self, py: Python<'_>, group: &PyAny) -> PyResult<PyGraphView> {
        let group = nonnegative_count(group, "group revision")?;
        py.allow_threads(|| self.inner.group_view(group))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn children(&self, py: Python<'_>, group: &PyAny) -> PyResult<Vec<usize>> {
        let group = nonnegative_count(group, "group revision")?;
        py.allow_threads(|| self.inner.children(group))
            .map_err(graph_error_to_py_err)
    }
    fn parents(&self, py: Python<'_>, group: &PyAny) -> PyResult<Vec<usize>> {
        let group = nonnegative_count(group, "group revision")?;
        py.allow_threads(|| self.inner.parents(group))
            .map_err(graph_error_to_py_err)
    }
    fn contains_node(&self, py: Python<'_>, group: &PyAny, node: &PyAny) -> PyResult<bool> {
        let group = nonnegative_count(group, "group revision")?;
        let node = nonnegative_count(node, "node")?;
        py.allow_threads(|| self.inner.contains_node(group, node))
            .map_err(graph_error_to_py_err)
    }
    fn contains_edge(&self, py: Python<'_>, group: &PyAny, edge: &PyAny) -> PyResult<bool> {
        let group = nonnegative_count(group, "group revision")?;
        let edge = nonnegative_count(edge, "edge")?;
        py.allow_threads(|| self.inner.contains_edge(group, edge))
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(group, *, recursive=false))]
    fn node_references(
        &self,
        py: Python<'_>,
        group: &PyAny,
        recursive: bool,
    ) -> PyResult<Vec<(usize, String)>> {
        let group = nonnegative_count(group, "group revision")?;
        py.allow_threads(|| self.inner.node_references(group, recursive))
            .map(|rows| {
                rows.into_iter()
                    .map(|row| (row.entity_id, row.property))
                    .collect()
            })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(group, *, recursive=false))]
    fn edge_references(
        &self,
        py: Python<'_>,
        group: &PyAny,
        recursive: bool,
    ) -> PyResult<Vec<(usize, String)>> {
        let group = nonnegative_count(group, "group revision")?;
        py.allow_threads(|| self.inner.edge_references(group, recursive))
            .map(|rows| {
                rows.into_iter()
                    .map(|row| (row.entity_id, row.property))
                    .collect()
            })
            .map_err(graph_error_to_py_err)
    }
    fn rebuild(&self, py: Python<'_>, view: &PyGraphView) -> PyResult<Self> {
        py.allow_threads(|| self.inner.rebuild(view.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}
