//! Translation of the Python batch description into the Rust write plan.

use crate::ffi::utils::python_value_to_attr_value;
use groggy::{BatchChangeRef, BatchEdgeRef, BatchNodeRef, GraphBatch};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn node_ref(value: &PyAny) -> PyResult<BatchNodeRef> {
    if let Ok(id) = value.extract::<usize>() {
        return Ok(BatchNodeRef::Existing(id));
    }
    if let Ok(reference) = value.downcast::<PyDict>() {
        if let Some(position) = reference.get_item("added")? {
            return Ok(BatchNodeRef::Added(position.extract()?));
        }
    }
    Err(PyTypeError::new_err(
        "node reference must be an existing ID or {'added': position}",
    ))
}

fn edge_ref(value: &PyAny) -> PyResult<BatchEdgeRef> {
    if let Ok(id) = value.extract::<usize>() {
        return Ok(BatchEdgeRef::Existing(id));
    }
    if let Ok(reference) = value.downcast::<PyDict>() {
        if let Some(position) = reference.get_item("added")? {
            return Ok(BatchEdgeRef::Added(position.extract()?));
        }
    }
    Err(PyTypeError::new_err(
        "edge reference must be an existing ID or {'added': position}",
    ))
}

fn pairs<'py>(value: &'py PyAny, context: &str) -> PyResult<Vec<(&'py PyAny, &'py PyAny)>> {
    let items = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err(format!("{context} must be a list of two-item pairs")))?;
    items
        .iter()
        .enumerate()
        .map(|(position, item)| {
            let pair = item.downcast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(format!("{context}[{position}] must be a pair"))
            })?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "{context}[{position}] must have exactly two items"
                )));
            }
            Ok((pair.get_item(0)?, pair.get_item(1)?))
        })
        .collect()
}

pub(crate) fn timestamp(value: &PyAny) -> PyResult<i64> {
    if value.is_instance_of::<pyo3::types::PyBool>()
        || !value.is_instance_of::<pyo3::types::PyLong>()
    {
        return Err(PyTypeError::new_err(
            "effective_at must be an integer of signed Unix microseconds",
        ));
    }
    value.extract()
}

pub(crate) fn parse_batch(batch: &PyAny) -> PyResult<GraphBatch> {
    let owned;
    let description = if let Ok(dict) = batch.downcast::<PyDict>() {
        dict
    } else {
        owned = batch.call_method0("to_dict")?;
        owned
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("GraphBatch.to_dict() must return a dictionary"))?
    };
    for key in description.keys().iter() {
        let key: String = key.extract()?;
        if ![
            "add_nodes",
            "add_edges",
            "node_attrs",
            "edge_attrs",
            "remove_nodes",
            "remove_edges",
            "effective_at",
            "change_times",
            "node_external_ids",
            "edge_external_ids",
        ]
        .contains(&key.as_str())
        {
            return Err(PyValueError::new_err(format!(
                "Unknown graph batch field {key}"
            )));
        }
    }
    let mut result = GraphBatch::default();
    if let Some(value) = description.get_item("effective_at")? {
        if !value.is_none() {
            result.effective_at = Some(timestamp(value)?);
        }
    }
    if let Some(times) = description.get_item("change_times")? {
        for time in times.downcast::<PyList>()?.iter() {
            let time = time.downcast::<PyDict>()?;
            let field = |name: &str| {
                time.get_item(name)?.ok_or_else(|| {
                    PyValueError::new_err(format!("change_times entry missing {name}"))
                })
            };
            let kind: String = field("kind")?.extract()?;
            let entity = field("entity")?;
            let key = match kind.as_str() {
                "node_added" => BatchChangeRef::NodeAdded(node_ref(entity)?),
                "node_removed" => BatchChangeRef::NodeRemoved(node_ref(entity)?),
                "edge_added" => BatchChangeRef::EdgeAdded(edge_ref(entity)?),
                "edge_removed" => BatchChangeRef::EdgeRemoved(edge_ref(entity)?),
                "node_attr" => {
                    BatchChangeRef::NodeAttribute(node_ref(entity)?, field("name")?.extract()?)
                }
                "edge_attr" => {
                    BatchChangeRef::EdgeAttribute(edge_ref(entity)?, field("name")?.extract()?)
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown change_times kind {kind}"
                    )))
                }
            };
            result
                .change_times
                .push((key, timestamp(field("effective_at")?)?));
        }
    }
    if let Some(rows) = description.get_item("node_external_ids")? {
        for (reference, name) in pairs(rows, "node_external_ids")? {
            result
                .node_external_ids
                .push((node_ref(reference)?, name.extract()?));
        }
    }
    if let Some(rows) = description.get_item("edge_external_ids")? {
        for (reference, name) in pairs(rows, "edge_external_ids")? {
            result
                .edge_external_ids
                .push((edge_ref(reference)?, name.extract()?));
        }
    }
    if let Some(count) = description.get_item("add_nodes")? {
        result.add_nodes = count.extract()?;
    }
    if let Some(edges) = description.get_item("add_edges")? {
        for (source, target) in pairs(edges, "add_edges")? {
            result
                .add_edges
                .push((node_ref(source)?, node_ref(target)?));
        }
    }
    if let Some(attrs) = description.get_item("node_attrs")? {
        for (name, values) in attrs.downcast::<PyDict>()?.iter() {
            let name: String = name.extract()?;
            let mut assignments = Vec::new();
            for (reference, value) in pairs(values, &format!("node_attrs.{name}"))? {
                assignments.push((node_ref(reference)?, python_value_to_attr_value(value)?));
            }
            result.node_attrs.insert(name, assignments);
        }
    }
    if let Some(attrs) = description.get_item("edge_attrs")? {
        for (name, values) in attrs.downcast::<PyDict>()?.iter() {
            let name: String = name.extract()?;
            let mut assignments = Vec::new();
            for (reference, value) in pairs(values, &format!("edge_attrs.{name}"))? {
                assignments.push((edge_ref(reference)?, python_value_to_attr_value(value)?));
            }
            result.edge_attrs.insert(name, assignments);
        }
    }
    if let Some(nodes) = description.get_item("remove_nodes")? {
        for reference in nodes.downcast::<PyList>()?.iter() {
            result.remove_nodes.push(node_ref(reference)?);
        }
    }
    if let Some(edges) = description.get_item("remove_edges")? {
        for reference in edges.downcast::<PyList>()?.iter() {
            result.remove_edges.push(edge_ref(reference)?);
        }
    }
    Ok(result)
}

/// Translate immutable core publication metadata without changing clock semantics.
pub(crate) fn publication_to_python(
    py: Python,
    metadata: &groggy::PublicationMetadata,
) -> PyResult<PyObject> {
    use groggy::ChangeKey;
    let output = PyDict::new(py);
    output.set_item("recorded_at", metadata.recorded_at)?;
    output.set_item("effective_at", metadata.effective_at)?;
    let changes = PyList::empty(py);
    for (key, at) in &metadata.change_times {
        let (kind, id, name) = match key {
            ChangeKey::NodeAdded(id) => ("node_added", *id, None),
            ChangeKey::NodeRemoved(id) => ("node_removed", *id, None),
            ChangeKey::EdgeAdded(id) => ("edge_added", *id, None),
            ChangeKey::EdgeRemoved(id) => ("edge_removed", *id, None),
            ChangeKey::NodeAttribute(id, name) => ("node_attr", *id, Some(name)),
            ChangeKey::EdgeAttribute(id, name) => ("edge_attr", *id, Some(name)),
        };
        let row = PyDict::new(py);
        row.set_item("kind", kind)?;
        row.set_item("entity_id", id)?;
        row.set_item("name", name)?;
        row.set_item("effective_at", at)?;
        changes.append(row)?;
    }
    output.set_item("change_times", changes)?;
    Ok(output.to_object(py))
}
