//! Thin typed conflict/answer translation. Native code owns policy, topology,
//! provenance, time validation and publication.
use super::graph_batch::{existing_id, state_id, timestamp};
use crate::ffi::utils::{
    attr_value_to_python_value, graph_error_to_py_err, python_value_to_attr_value,
};
use groggy::state::{
    MergeEntity, MergeProperties, SnapshotMergeConflict as C, SnapshotMergeResolution as R,
    SnapshotMergeSide,
};
use groggy::{AttrValue, ChangeKey, GraphMergeOptions, GraphMergeOutcome, GraphMergePlan};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};
fn field<'a>(d: &'a PyDict, key: &str) -> PyResult<&'a PyAny> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing merge field {key}")))
}
fn rows(py: Python, values: &MergeProperties) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    for (name, value) in values {
        d.set_item(name, attr_value_to_python_value(py, value)?)?;
    }
    Ok(d.into())
}
fn entity(py: Python, value: &Option<MergeEntity>) -> PyResult<PyObject> {
    let Some(value) = value else {
        return Ok(py.None());
    };
    let d = PyDict::new(py);
    d.set_item("endpoints", value.endpoints)?;
    d.set_item("properties", rows(py, &value.properties)?)?;
    Ok(d.into())
}
fn slot(py: Python, value: &Option<AttrValue>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("present", value.is_some())?;
    if let Some(value) = value {
        d.set_item("value", attr_value_to_python_value(py, value)?)?;
        d.set_item("dtype", value.type_name())?;
    }
    Ok(d.into())
}
fn conflict_to_py(py: Python, c: &C) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    match c {
        C::Node {
            id,
            base,
            left,
            right,
        }
        | C::Edge {
            id,
            base,
            left,
            right,
        } => {
            d.set_item(
                "kind",
                if matches!(c, C::Node { .. }) {
                    "node"
                } else {
                    "edge"
                },
            )?;
            d.set_item("entity", id)?;
            d.set_item("base", entity(py, base)?)?;
            d.set_item("left", entity(py, left)?)?;
            d.set_item("right", entity(py, right)?)?;
        }
        C::NodeProperty {
            id,
            name,
            base,
            left,
            right,
        }
        | C::EdgeProperty {
            id,
            name,
            base,
            left,
            right,
        } => {
            d.set_item(
                "kind",
                if matches!(c, C::NodeProperty { .. }) {
                    "node_attr"
                } else {
                    "edge_attr"
                },
            )?;
            d.set_item("entity", id)?;
            d.set_item("name", name)?;
            d.set_item("base", slot(py, base)?)?;
            d.set_item("left", slot(py, left)?)?;
            d.set_item("right", slot(py, right)?)?;
        }
        C::EdgeEndpoints {
            id,
            base,
            left,
            right,
        } => {
            d.set_item("kind", "edge_endpoints")?;
            d.set_item("entity", id)?;
            d.set_item("base", base)?;
            d.set_item("left", left)?;
            d.set_item("right", right)?;
        }
        C::Topology { id, endpoints } => {
            d.set_item("kind", "topology")?;
            d.set_item("entity", id)?;
            d.set_item("endpoints", endpoints)?;
        }
    }
    Ok(d.into())
}
fn pair(value: &PyAny) -> PyResult<(usize, usize)> {
    let v: Vec<&PyAny> = value.extract()?;
    if v.len() != 2 {
        return Err(PyValueError::new_err("Endpoints require exactly two IDs"));
    }
    Ok((existing_id(v[0])?, existing_id(v[1])?))
}
fn properties(value: &PyAny) -> PyResult<MergeProperties> {
    value
        .downcast::<PyDict>()?
        .iter()
        .map(|(name, value)| Ok((name.extract()?, python_value_to_attr_value(value)?)))
        .collect()
}
fn parse_entity(value: &PyAny) -> PyResult<Option<MergeEntity>> {
    if value.is_none() {
        return Ok(None);
    }
    let d = value.downcast::<PyDict>()?;
    let endpoints = d
        .get_item("endpoints")?
        .filter(|v| !v.is_none())
        .map(pair)
        .transpose()?;
    Ok(Some(MergeEntity {
        endpoints,
        properties: properties(field(d, "properties")?)?,
    }))
}
fn answer(c: &C, value: &PyAny) -> PyResult<R> {
    if let Ok(side) = value.extract::<String>() {
        let side = match side.as_str() {
            "base" => SnapshotMergeSide::Base,
            "left" => SnapshotMergeSide::Left,
            "right" => SnapshotMergeSide::Right,
            _ => {
                return Err(PyValueError::new_err(
                    "Merge side must be base, left or right",
                ))
            }
        };
        return c.resolution_for(side).map_err(graph_error_to_py_err);
    }
    let d = value.downcast::<PyDict>()?;
    let kind: String = field(d, "kind")?.extract()?;
    match kind.as_str() {
        "entity" => Ok(R::Entity(parse_entity(field(d, "value")?)?)),
        "property" => {
            let present: bool = field(d, "present")?.extract()?;
            if !present && d.get_item("value")?.is_some() {
                return Err(PyValueError::new_err(
                    "Absent property answer cannot include value",
                ));
            }
            Ok(R::Property(if present {
                Some(python_value_to_attr_value(field(d, "value")?)?)
            } else {
                None
            }))
        }
        "endpoints" => Ok(R::Endpoints(pair(field(d, "value")?)?)),
        "drop_edge" => Ok(R::DropEdge),
        "endpoint_nodes" => {
            let records = field(d, "nodes")?.downcast::<PyList>()?;
            let mut result = Vec::new();
            for record in records.iter() {
                let record = record.downcast::<PyDict>()?;
                result.push((
                    existing_id(field(record, "id")?)?,
                    properties(field(record, "properties")?)?,
                ));
            }
            Ok(R::EndpointNodes(result))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown merge answer kind {kind}"
        ))),
    }
}
pub(crate) fn options(
    effective_at: Option<&PyAny>,
    change_times: Option<&PyList>,
) -> PyResult<GraphMergeOptions> {
    let mut options = GraphMergeOptions {
        effective_at: effective_at.map(timestamp).transpose()?,
        ..Default::default()
    };
    if let Some(times) = change_times {
        for row in times.iter() {
            let d = row.downcast::<PyDict>()?;
            let kind: String = field(d, "kind")?.extract()?;
            let id = existing_id(field(d, "entity")?)?;
            let key = match kind.as_str() {
                "node_added" => ChangeKey::NodeAdded(id),
                "node_removed" => ChangeKey::NodeRemoved(id),
                "edge_added" => ChangeKey::EdgeAdded(id),
                "edge_removed" => ChangeKey::EdgeRemoved(id),
                "edge_endpoints" => ChangeKey::EdgeEndpoints(id),
                "node_attr" => ChangeKey::NodeAttribute(id, field(d, "name")?.extract()?),
                "edge_attr" => ChangeKey::EdgeAttribute(id, field(d, "name")?.extract()?),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown merge change-time kind {kind}"
                    )))
                }
            };
            if options
                .change_times
                .insert(key, timestamp(field(d, "effective_at")?)?)
                .is_some()
            {
                return Err(PyValueError::new_err(
                    "Duplicate merge change-time override",
                ));
            }
        }
    }
    Ok(options)
}
pub(crate) fn selections(
    right: &PyAny,
    base: Option<&PyAny>,
    expected: Option<&PyAny>,
) -> PyResult<(u64, Option<u64>, Option<u64>)> {
    Ok((
        state_id(right)?,
        base.map(state_id).transpose()?,
        expected.map(state_id).transpose()?,
    ))
}
#[pyclass(name = "MergePlan")]
pub struct PyMergePlan {
    pub(crate) inner: Option<GraphMergePlan>,
}
#[pymethods]
impl PyMergePlan {
    #[getter]
    fn is_consumed(&self) -> bool {
        self.inner.is_none()
    }
    #[getter]
    fn parents(&self) -> PyResult<[u64; 2]> {
        Ok(self.plan()?.parents())
    }
    #[getter]
    fn base_state(&self) -> PyResult<u64> {
        Ok(self.plan()?.base_state())
    }
    #[getter]
    fn conflicts(&self, py: Python) -> PyResult<PyObject> {
        let values = self
            .plan()?
            .conflicts()
            .iter()
            .map(|c| conflict_to_py(py, c))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, values).into())
    }
    #[pyo3(signature=(resolver=None))]
    fn resolve(&mut self, py: Python, resolver: Option<PyObject>) -> PyResult<PyMergeOutcome> {
        if let Some(resolver) = &resolver {
            if !resolver.as_ref(py).is_callable() {
                return Err(PyTypeError::new_err("Merge resolver must be callable"));
            }
        }
        let plan = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("MergePlan already consumed"))?;
        let mut original_error = None;
        let result = py.allow_threads(|| match resolver {
            None => plan.into_outcome(),
            Some(resolver) => plan.resolve_with(|c| {
                Python::with_gil(|py| {
                    let value = resolver.call1(py, (conflict_to_py(py, c)?,))?;
                    answer(c, value.as_ref(py))
                })
                .map_err(|error| {
                    let message = error.to_string();
                    original_error = Some(error);
                    groggy::GraphError::InvalidInput(message)
                })
            }),
        });
        if let Some(error) = original_error {
            return Err(error);
        }
        Ok(PyMergeOutcome {
            inner: Some(result.map_err(graph_error_to_py_err)?),
        })
    }
}
impl PyMergePlan {
    fn plan(&self) -> PyResult<&GraphMergePlan> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MergePlan already consumed"))
    }
}
#[pyclass(name = "MergeOutcome")]
pub struct PyMergeOutcome {
    inner: Option<GraphMergeOutcome>,
}
#[pymethods]
impl PyMergeOutcome {
    #[getter]
    fn is_consumed(&self) -> bool {
        self.inner.is_none()
    }
    #[getter]
    fn parents(&self) -> PyResult<[u64; 2]> {
        Ok(self.outcome()?.parents())
    }
    #[getter]
    fn base_state(&self) -> PyResult<u64> {
        Ok(self.outcome()?.base_state())
    }
    #[getter]
    fn branch(&self) -> PyResult<String> {
        Ok(self.outcome()?.branch().into())
    }
}
impl PyMergeOutcome {
    fn outcome(&self) -> PyResult<&GraphMergeOutcome> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MergeOutcome already consumed"))
    }
    pub(crate) fn take(&mut self) -> PyResult<GraphMergeOutcome> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("MergeOutcome already consumed"))
    }
}
