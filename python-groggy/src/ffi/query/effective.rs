//! Detached effective-time indexing; only argument/result translation lives here.
use crate::ffi::{
    api::graph_batch::{change_key_to_python, timestamp},
    subgraphs::view::PyGraphView,
    utils::{attr_value_to_python_value, graph_error_to_py_err},
};
use groggy::query::{EffectiveTimeEvent, EffectiveTimeIndex, PropertyInterval};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
fn cutoff(value: Option<&PyAny>) -> PyResult<Option<i64>> {
    value.map(timestamp).transpose()
}
fn rows(py: Python<'_>, events: Vec<EffectiveTimeEvent>) -> PyResult<PyObject> {
    let rows = PyList::empty(py);
    for event in events {
        let row = change_key_to_python(py, &event.change)?;
        row.set_item("state_id", event.state_id)?;
        row.set_item("effective_at", event.effective_at)?;
        row.set_item("recorded_at", event.recorded_at)?;
        rows.append(row)?;
    }
    Ok(rows.into())
}
fn intervals(py: Python<'_>, intervals: Vec<PropertyInterval>) -> PyResult<PyObject> {
    let rows = PyList::empty(py);
    for interval in intervals {
        let row = PyDict::new(py);
        row.set_item("start", interval.start)?;
        row.set_item("end", interval.end)?;
        row.set_item("present", interval.value.is_some())?;
        let value = match interval.value {
            Some(value) => attr_value_to_python_value(py, &value)?,
            None => py.None(),
        };
        row.set_item("value", value)?;
        rows.append(row)?;
    }
    Ok(rows.into())
}
#[pyclass(name = "EffectiveTimeIndex")]
#[derive(Clone)]
pub struct PyEffectiveTimeIndex {
    pub inner: EffectiveTimeIndex,
}
#[pymethods]
impl PyEffectiveTimeIndex {
    #[getter]
    fn branch(&self) -> String {
        self.inner.branch().into()
    }
    #[getter]
    fn captured_head(&self) -> u64 {
        self.inner.captured_head()
    }
    #[getter]
    fn event_count(&self) -> usize {
        self.inner.event_count()
    }
    #[pyo3(signature=(node, name, *, recorded_by=None))]
    fn node_property_intervals(
        &self,
        py: Python<'_>,
        node: &PyAny,
        name: &str,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let id = super::query::nonnegative_count(node, "node")?;
        let by = cutoff(recorded_by)?;
        intervals(
            py,
            py.allow_threads(|| self.inner.node_property_intervals(id, name, by))
                .map_err(graph_error_to_py_err)?,
        )
    }
    #[pyo3(signature=(edge, name, *, recorded_by=None))]
    fn edge_property_intervals(
        &self,
        py: Python<'_>,
        edge: &PyAny,
        name: &str,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let id = super::query::nonnegative_count(edge, "edge")?;
        let by = cutoff(recorded_by)?;
        intervals(
            py,
            py.allow_threads(|| self.inner.edge_property_intervals(id, name, by))
                .map_err(graph_error_to_py_err)?,
        )
    }
    #[pyo3(signature=(effective_at, *, recorded_by=None))]
    fn read_view_at(
        &self,
        py: Python<'_>,
        effective_at: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyGraphView> {
        let at = timestamp(effective_at)?;
        let by = cutoff(recorded_by)?;
        py.allow_threads(|| self.inner.read_view_at(at, by))
            .map(|inner| PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(start, end, *, recorded_by=None))]
    fn events_between(
        &self,
        py: Python<'_>,
        start: &PyAny,
        end: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let start = timestamp(start)?;
        let end = timestamp(end)?;
        let by = cutoff(recorded_by)?;
        rows(
            py,
            py.allow_threads(|| self.inner.events_between(start, end, by))
                .map_err(graph_error_to_py_err)?,
        )
    }
    #[pyo3(signature=(node, start, end, *, recorded_by=None))]
    fn node_events(
        &self,
        py: Python<'_>,
        node: &PyAny,
        start: &PyAny,
        end: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let id = super::query::nonnegative_count(node, "node")?;
        let start = timestamp(start)?;
        let end = timestamp(end)?;
        let by = cutoff(recorded_by)?;
        rows(
            py,
            py.allow_threads(|| self.inner.node_events(id, start, end, by))
                .map_err(graph_error_to_py_err)?,
        )
    }
    #[pyo3(signature=(edge, start, end, *, recorded_by=None))]
    fn edge_events(
        &self,
        py: Python<'_>,
        edge: &PyAny,
        start: &PyAny,
        end: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let id = super::query::nonnegative_count(edge, "edge")?;
        let start = timestamp(start)?;
        let end = timestamp(end)?;
        let by = cutoff(recorded_by)?;
        rows(
            py,
            py.allow_threads(|| self.inner.edge_events(id, start, end, by))
                .map_err(graph_error_to_py_err)?,
        )
    }
    #[pyo3(signature=(node, *, recorded_by=None))]
    fn node_activation_intervals(
        &self,
        py: Python<'_>,
        node: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<Vec<(i64, Option<i64>)>> {
        let id = super::query::nonnegative_count(node, "node")?;
        let by = cutoff(recorded_by)?;
        Ok(py
            .allow_threads(|| self.inner.node_activation_intervals(id, by))
            .into_iter()
            .map(|row| (row.start, row.end))
            .collect())
    }
    #[pyo3(signature=(edge, *, recorded_by=None))]
    fn edge_activation_intervals(
        &self,
        py: Python<'_>,
        edge: &PyAny,
        recorded_by: Option<&PyAny>,
    ) -> PyResult<Vec<(i64, Option<i64>)>> {
        let id = super::query::nonnegative_count(edge, "edge")?;
        let by = cutoff(recorded_by)?;
        Ok(py
            .allow_threads(|| self.inner.edge_activation_intervals(id, by))
            .into_iter()
            .map(|row| (row.start, row.end))
            .collect())
    }
    fn copy(&self) -> Self {
        self.clone()
    }
}
