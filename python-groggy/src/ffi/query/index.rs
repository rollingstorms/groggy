//! Translation and detached native calls only; index semantics live in Rust.
use super::query::{edge_filter_from_python, node_filter_from_python};
use crate::ffi::{subgraphs::view::PyGraphView, utils::graph_error_to_py_err};
use groggy::query::GraphQueryIndex;
use pyo3::prelude::*;

#[pyclass(name = "GraphQueryIndex")]
#[derive(Clone)]
pub struct PyGraphQueryIndex {
    pub inner: GraphQueryIndex,
}
#[pymethods]
impl PyGraphQueryIndex {
    #[getter]
    fn view(&self) -> PyGraphView {
        PyGraphView {
            inner: self.inner.view().clone(),
        }
    }
    #[getter]
    fn node_properties(&self) -> Vec<String> {
        self.inner.node_properties()
    }
    #[getter]
    fn edge_properties(&self) -> Vec<String> {
        self.inner.edge_properties()
    }
    #[getter]
    fn has_topology(&self) -> bool {
        self.inner.has_topology()
    }
    fn rebuild(&self, py: Python<'_>, view: &PyGraphView) -> PyResult<Self> {
        py.allow_threads(|| self.inner.rebuild(view.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(filter, *, allow_scan=false))]
    fn find_nodes(&self, py: Python<'_>, filter: &PyAny, allow_scan: bool) -> PyResult<Vec<usize>> {
        let filter = node_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.find_nodes(&filter, allow_scan))
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(filter, *, allow_scan=false))]
    fn find_edges(&self, py: Python<'_>, filter: &PyAny, allow_scan: bool) -> PyResult<Vec<usize>> {
        let filter = edge_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.find_edges(&filter, allow_scan))
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(filter, *, allow_scan=false))]
    fn filter_nodes(
        &self,
        py: Python<'_>,
        filter: &PyAny,
        allow_scan: bool,
    ) -> PyResult<PyGraphView> {
        let filter = node_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.filter_nodes(&filter, allow_scan))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(filter, *, allow_scan=false))]
    fn filter_edges(
        &self,
        py: Python<'_>,
        filter: &PyAny,
        allow_scan: bool,
    ) -> PyResult<PyGraphView> {
        let filter = edge_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.filter_edges(&filter, allow_scan))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn nodes_with_label(&self, py: Python<'_>, label: &str) -> PyResult<PyGraphView> {
        py.allow_threads(|| self.inner.nodes_with_label(label))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}
