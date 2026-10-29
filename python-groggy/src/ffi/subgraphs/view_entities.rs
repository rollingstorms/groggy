//! Translation-only read accessors over canonical immutable entity adapters.
use super::view::PyGraphView;
use crate::ffi::{
    storage::{array::PyBaseArray, num_array::PyIntArray},
    utils::{attr_value_to_python_value, graph_error_to_py_err},
};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PySlice},
};

#[derive(Clone)]
#[pyclass(name = "GraphViewEntity", frozen)]
pub struct PyGraphViewEntity {
    pub inner: groggy::subgraphs::GraphViewEntity,
}
#[pymethods]
impl PyGraphViewEntity {
    #[getter]
    fn id(&self) -> usize {
        self.inner.id()
    }
    #[getter]
    fn kind(&self) -> &str {
        if self.inner.is_node() {
            "node"
        } else {
            "edge"
        }
    }
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match py.allow_threads(|| self.inner.get(&name.into())) {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Err(pyo3::exceptions::PyKeyError::new_err(name.to_owned())),
        }
    }
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match py.allow_threads(|| self.inner.get(&name.into())) {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(name.to_owned())),
        }
    }
    fn __contains__(&self, name: &str) -> bool {
        self.inner.contains_attribute(&name.into())
    }
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let attrs = py.allow_threads(|| self.inner.to_dict());
        let result = PyDict::new(py);
        for (name, value) in attrs {
            result.set_item(name, attr_value_to_python_value(py, &value)?)?;
        }
        Ok(result.to_object(py))
    }
    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.degree())
            .map_err(graph_error_to_py_err)
    }
    #[getter]
    fn neighbors(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        py.allow_threads(|| self.inner.neighbors())
            .map_err(graph_error_to_py_err)
    }
    #[getter]
    fn source(&self) -> PyResult<usize> {
        self.inner
            .endpoints()
            .map(|(s, _)| s)
            .map_err(graph_error_to_py_err)
    }
    #[getter]
    fn target(&self) -> PyResult<usize> {
        self.inner
            .endpoints()
            .map(|(_, t)| t)
            .map_err(graph_error_to_py_err)
    }
    #[getter]
    fn endpoints(&self) -> PyResult<(usize, usize)> {
        self.inner.endpoints().map_err(graph_error_to_py_err)
    }
    fn meta_subgraph(&self, py: Python<'_>) -> PyResult<PyGraphView> {
        py.allow_threads(|| self.inner.meta_subgraph())
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
}

#[derive(Clone)]
#[pyclass(name = "GraphViewEntities", frozen)]
pub struct PyGraphViewEntities {
    pub inner: groggy::subgraphs::GraphViewEntities,
}
#[pymethods]
impl PyGraphViewEntities {
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn kind(&self) -> &str {
        if self.inner.is_nodes() {
            "nodes"
        } else {
            "edges"
        }
    }
    #[getter]
    fn ids(&self, py: Python<'_>) -> PyResult<PyIntArray> {
        PyIntArray::from_node_ids(py.allow_threads(|| self.inner.ids().to_vec()))
    }
    fn attribute_names(&self, py: Python<'_>) -> Vec<String> {
        py.allow_threads(|| self.inner.attribute_names())
    }
    fn column(&self, py: Python<'_>, name: String) -> PyBaseArray {
        PyBaseArray {
            inner: py.allow_threads(|| self.inner.column(&name)),
        }
    }
    fn presence(&self, py: Python<'_>, name: String) -> PyBaseArray {
        PyBaseArray {
            inner: py.allow_threads(|| self.inner.presence(&name)),
        }
    }
    fn __contains__(&self, id: &PyAny) -> PyResult<bool> {
        let id = crate::ffi::query::query::nonnegative_count(id, "entity ID")?;
        Ok(self.inner.contains_id(id))
    }
    fn __getattr__(&self, py: Python<'_>, name: String) -> PyResult<PyBaseArray> {
        if !py.allow_threads(|| self.inner.has_column(&name)) {
            return Err(pyo3::exceptions::PyAttributeError::new_err(name));
        }
        Ok(self.column(py, name))
    }
    fn by_id(&self, id: &PyAny) -> PyResult<PyGraphViewEntity> {
        let id = crate::ffi::query::query::nonnegative_count(id, "entity ID")?;
        self.inner
            .by_id(id)
            .map(|inner| PyGraphViewEntity { inner })
            .map_err(graph_error_to_py_err)
    }
    fn at(&self, py: Python<'_>, position: &PyAny) -> PyResult<PyGraphViewEntity> {
        if position.is_instance_of::<PyBool>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "position must be an integer",
            ));
        }
        let position = position.extract::<isize>()?;
        py.allow_threads(|| self.inner.at(position))
            .map(|inner| PyGraphViewEntity { inner })
            .map_err(|error| pyo3::exceptions::PyIndexError::new_err(error.to_string()))
    }
    fn all(&self, py: Python<'_>) -> PyResult<PyGraphView> {
        py.allow_threads(|| self.inner.all())
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn select(&self, py: Python<'_>, ids: &PyAny) -> PyResult<PyGraphView> {
        let ids = ids
            .iter()?
            .map(|id| {
                id.and_then(|id| crate::ffi::query::query::nonnegative_count(id, "entity ID"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.allow_threads(|| self.inner.select(&ids))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn filter(&self, py: Python<'_>, filter: &PyAny) -> PyResult<PyGraphView> {
        let result = if self.inner.is_nodes() {
            let filter = crate::ffi::query::query::node_filter_from_python(filter)?;
            py.allow_threads(|| self.inner.filter_nodes(&filter))
        } else {
            let filter = crate::ffi::query::query::edge_filter_from_python(filter)?;
            py.allow_threads(|| self.inner.filter_edges(&filter))
        };
        result
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    fn __getitem__(&self, py: Python<'_>, key: &PyAny) -> PyResult<PyObject> {
        if let Ok(name) = key.extract::<String>() {
            return Ok(Py::new(py, self.column(py, name))?.to_object(py));
        }
        if let Ok(mask) = key.extract::<PyRef<PyBaseArray>>() {
            let mask = mask.inner.clone();
            let view = py
                .allow_threads(|| self.inner.select_mask(&mask))
                .map_err(graph_error_to_py_err)?;
            return Ok(Py::new(py, PyGraphView { inner: view })?.to_object(py));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let length = i64::try_from(self.inner.len()).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "collection is too large for Python slicing",
                )
            })?;
            let indices = slice.indices(length)?;
            let view = py
                .allow_threads(|| {
                    self.inner.slice_positions(
                        indices.start,
                        indices.step,
                        indices.slicelength as usize,
                    )
                })
                .map_err(graph_error_to_py_err)?;
            return Ok(Py::new(py, PyGraphView { inner: view })?.to_object(py));
        }
        if key.downcast::<pyo3::types::PyList>().is_ok()
            || key.downcast::<pyo3::types::PyTuple>().is_ok()
        {
            return Ok(Py::new(py, self.select(py, key)?)?.to_object(py));
        }
        Ok(Py::new(py, self.by_id(key)?)?.to_object(py))
    }
    fn __iter__(&self, py: Python<'_>) -> PyGraphViewEntityIterator {
        PyGraphViewEntityIterator {
            inner: py.allow_threads(|| self.inner.iter()),
        }
    }
}
#[pyclass(name = "GraphViewEntityIterator")]
pub struct PyGraphViewEntityIterator {
    inner: groggy::subgraphs::GraphViewEntityIterator,
}
#[pymethods]
impl PyGraphViewEntityIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self) -> Option<PyGraphViewEntity> {
        self.inner.next().map(|inner| PyGraphViewEntity { inner })
    }
    fn __length_hint__(&self) -> usize {
        self.inner.len()
    }
}
