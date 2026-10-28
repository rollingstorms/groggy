//! Strict pattern-shape marshalling; all matching and search policy is native.
use super::query::{edge_filter_from_python, node_filter_from_python};
use crate::ffi::{subgraphs::view::PyGraphView, utils::graph_error_to_py_err};
use groggy::query::{GraphPattern, PatternEdge, PatternMatches, PatternNode};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};

fn check_fields(row: &PyDict, allowed: &[&str]) -> PyResult<()> {
    for (key, _) in row.iter() {
        let key = key.extract::<String>()?;
        if !allowed.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Unknown pattern field: {key}"
            )));
        }
    }
    Ok(())
}
fn text(row: &PyDict, name: &str) -> PyResult<String> {
    row.get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing pattern field: {name}")))?
        .extract()
}
#[pyclass(name = "GraphPattern")]
#[derive(Clone)]
pub struct PyGraphPattern {
    pub inner: GraphPattern,
}
#[pymethods]
impl PyGraphPattern {
    #[new]
    #[pyo3(signature=(*, nodes, edges=None))]
    fn new(nodes: &PyList, edges: Option<&PyList>) -> PyResult<Self> {
        let nodes = nodes
            .iter()
            .map(|row| {
                let row = row.downcast::<PyDict>()?;
                check_fields(row, &["name", "filter"])?;
                Ok(PatternNode {
                    variable: text(row, "name")?,
                    filter: row
                        .get_item("filter")?
                        .filter(|v| !v.is_none())
                        .map(node_filter_from_python)
                        .transpose()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut parsed_edges = Vec::new();
        if let Some(edges) = edges {
            for row in edges.iter() {
                let row = row.downcast::<PyDict>()?;
                check_fields(row, &["name", "source", "target", "filter", "direction"])?;
                let direction = row
                    .get_item("direction")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| "outgoing".into());
                parsed_edges.push(PatternEdge {
                    variable: text(row, "name")?,
                    source: text(row, "source")?,
                    target: text(row, "target")?,
                    direction: direction.parse().map_err(graph_error_to_py_err)?,
                    filter: row
                        .get_item("filter")?
                        .filter(|v| !v.is_none())
                        .map(edge_filter_from_python)
                        .transpose()?,
                });
            }
        }
        let inner = GraphPattern {
            nodes,
            edges: parsed_edges,
        };
        inner.validate().map_err(graph_error_to_py_err)?;
        Ok(Self { inner })
    }
    #[getter]
    fn node_variables(&self) -> Vec<String> {
        self.inner
            .nodes
            .iter()
            .map(|v| v.variable.clone())
            .collect()
    }
    #[getter]
    fn edge_variables(&self) -> Vec<String> {
        self.inner
            .edges
            .iter()
            .map(|v| v.variable.clone())
            .collect()
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}
#[pyclass(name = "PatternMatches")]
#[derive(Clone)]
pub struct PyPatternMatches {
    pub inner: PatternMatches,
}
#[pymethods]
impl PyPatternMatches {
    fn __len__(&self) -> usize {
        self.inner.rows().len()
    }
    #[getter]
    fn source_view(&self) -> PyGraphView {
        PyGraphView {
            inner: self.inner.source_view().clone(),
        }
    }
    #[getter]
    fn node_variables(&self) -> Vec<String> {
        self.inner.node_variables().to_vec()
    }
    #[getter]
    fn edge_variables(&self) -> Vec<String> {
        self.inner.edge_variables().to_vec()
    }
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated()
    }
    #[getter]
    fn expansions(&self) -> usize {
        self.inner.expansions()
    }
    #[getter]
    fn bindings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for row in self.inner.rows() {
            let nodes = PyDict::new(py);
            for (name, id) in self.inner.node_variables().iter().zip(&row.nodes) {
                nodes.set_item(name, id)?;
            }
            let edges = PyDict::new(py);
            for (name, id) in self.inner.edge_variables().iter().zip(&row.edges) {
                edges.set_item(name, id)?;
            }
            let binding = PyDict::new(py);
            binding.set_item("nodes", nodes)?;
            binding.set_item("edges", edges)?;
            result.append(binding)?;
        }
        Ok(result.into())
    }
    fn match_view(&self, py: Python<'_>, row: &PyAny) -> PyResult<PyGraphView> {
        let row = super::query::nonnegative_count(row, "match row")?;
        py.allow_threads(|| self.inner.match_view(row))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}
