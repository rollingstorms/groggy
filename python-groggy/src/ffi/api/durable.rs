//! Thin translation for the binary journal and its checked writer.
use super::graph::{parse_config, parse_edge_upserts, parse_node_upserts, parse_schema};
use super::graph_batch::{parse_batch, publication_to_python, timestamp};
use crate::ffi::temporal::PyTemporalSnapshot;
use crate::ffi::utils::{attr_value_to_python_value, graph_error_to_py_err};
use groggy::{DurableGraph, StateId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

#[pyclass(name = "DurableGraph", unsendable)]
pub struct PyDurableGraph {
    store: Option<DurableGraph>,
}
impl PyDurableGraph {
    fn core(&self) -> PyResult<&DurableGraph> {
        self.store
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("DurableGraph is closed"))
    }
    pub(crate) fn core_mut(&mut self) -> PyResult<&mut DurableGraph> {
        self.store
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("DurableGraph is closed"))
    }
}
#[pymethods]
impl PyDurableGraph {
    #[staticmethod]
    #[pyo3(signature=(path, *, directed=None, config=None, schema=None))]
    fn create(
        path: PathBuf,
        directed: Option<bool>,
        config: Option<&PyDict>,
        schema: Option<&PyDict>,
    ) -> PyResult<Self> {
        let config = parse_config(config, directed)?;
        let schema = schema.map(parse_schema).transpose()?;
        Ok(Self {
            store: Some(
                DurableGraph::create(&path, config, schema).map_err(graph_error_to_py_err)?,
            ),
        })
    }
    /// Begin a journal from an existing graph without changing its owner.
    #[staticmethod]
    #[pyo3(signature=(path, graph, *, schema=None))]
    fn create_from_graph(
        path: PathBuf,
        graph: &super::graph::PyGraph,
        schema: Option<&PyDict>,
    ) -> PyResult<Self> {
        let schema = schema.map(parse_schema).transpose()?;
        Ok(Self {
            store: Some(
                DurableGraph::create_from_graph(&path, &graph.inner.borrow(), schema)
                    .map_err(graph_error_to_py_err)?,
            ),
        })
    }
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            store: Some(DurableGraph::open(&path).map_err(graph_error_to_py_err)?),
        })
    }
    fn backup(&mut self, destination: PathBuf) -> PyResult<()> {
        self.core_mut()?
            .backup(&destination)
            .map_err(graph_error_to_py_err)
    }
    /// Explicit physical journal baseline; does not publish a graph version.
    #[pyo3(signature=(*,expected_state=None))]
    fn checkpoint_storage(
        &mut self,
        py: Python<'_>,
        expected_state: Option<StateId>,
    ) -> PyResult<PyObject> {
        let expected = expected_state.unwrap_or(self.current_state_id()?);
        let plan = self
            .core_mut()?
            .prepare_storage_checkpoint(expected)
            .map_err(graph_error_to_py_err)?;
        let outcome = py.allow_threads(move || plan.execute());
        let report = self
            .core_mut()?
            .finish_storage_checkpoint(outcome)
            .map_err(graph_error_to_py_err)?;
        storage_checkpoint_report_to_py(py, &report)
    }
    /// Explicit exact-value pool compaction and verified physical replacement.
    #[pyo3(signature=(*,expected_state=None))]
    fn compact_storage(
        &mut self,
        py: Python<'_>,
        expected_state: Option<StateId>,
    ) -> PyResult<PyObject> {
        let expected = expected_state.unwrap_or(self.current_state_id()?);
        let plan = self
            .core_mut()?
            .prepare_storage_compaction(expected)
            .map_err(graph_error_to_py_err)?;
        let outcome = py.allow_threads(move || plan.execute());
        let report = self
            .core_mut()?
            .finish_storage_checkpoint(outcome)
            .map_err(graph_error_to_py_err)?;
        storage_checkpoint_report_to_py(py, &report)
    }
    fn close(&mut self) {
        self.store = None;
    }
    #[getter]
    fn is_closed(&self) -> bool {
        self.store.is_none()
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.core()?;
        Ok(slf)
    }
    fn __exit__(&mut self, _kind: &PyAny, _value: &PyAny, _traceback: &PyAny) -> bool {
        self.store = None;
        false
    }
    #[getter]
    fn current_state_id(&self) -> PyResult<StateId> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .current_state_id())
    }
    #[getter]
    fn schema_revision(&self) -> PyResult<u64> {
        Ok(self.core()?.schema_revision())
    }
    #[getter]
    fn recovered_tail_bytes(&self) -> PyResult<u64> {
        Ok(self.core()?.recovered_tail_bytes())
    }
    #[getter]
    fn is_directed(&self) -> PyResult<bool> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .is_directed())
    }
    fn has_uncommitted_changes(&self) -> PyResult<bool> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .has_uncommitted_changes())
    }
    fn node_count(&self) -> PyResult<usize> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .node_count())
    }
    fn edge_count(&self) -> PyResult<usize> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .edge_count())
    }
    fn node_ids(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .node_ids())
    }
    fn edge_ids(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .edge_ids())
    }
    fn get_node_attr(&self, py: Python, node: usize, name: String) -> PyResult<PyObject> {
        match self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .get_node_attr(node, &name)
            .map_err(graph_error_to_py_err)?
        {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Ok(py.None()),
        }
    }
    fn get_edge_attr(&self, py: Python, edge: usize, name: String) -> PyResult<PyObject> {
        match self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .get_edge_attr(edge, &name)
            .map_err(graph_error_to_py_err)?
        {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Ok(py.None()),
        }
    }
    #[pyo3(signature=(external_id,group_id=None))]
    fn node_by_external_id(
        &self,
        external_id: &str,
        group_id: Option<&str>,
    ) -> PyResult<Option<usize>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .node_by_external_id(external_id, group_id))
    }
    #[pyo3(signature=(external_id,group_id=None))]
    fn edge_by_external_id(
        &self,
        external_id: &str,
        group_id: Option<&str>,
    ) -> PyResult<Option<usize>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .edge_by_external_id(external_id, group_id))
    }
    fn allocated_entity_for_external_id(
        &self,
        name: &str,
    ) -> PyResult<Option<(&'static str, usize)>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .allocated_entity_for_external_id(name)
            .map(|entity| match entity {
                groggy::ExternalEntityRef::Node(id) => ("node", id),
                groggy::ExternalEntityRef::Edge(id) => ("edge", id),
            }))
    }
    #[pyo3(signature=(*,author="writer",schema=None))]
    fn writer(
        slf: PyRef<'_, Self>,
        py: Python,
        author: &str,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        slf.core()?;
        if let Some(schema) = schema {
            parse_schema(schema)?;
        }
        Ok(py
            .import("groggy.writer")?
            .getattr("GraphWriter")?
            .call1((slf.into_py(py), author, schema))?
            .to_object(py))
    }
    #[pyo3(signature=(batch,expected_state,message,author,*,schema=None))]
    fn apply_batch(
        &mut self,
        py: Python,
        batch: &PyAny,
        expected_state: StateId,
        message: String,
        author: String,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let batch = parse_batch(batch)?;
        let schema = schema.map(parse_schema).transpose()?;
        let result = self
            .core_mut()?
            .apply_batch_with_schema(batch, expected_state, message, author, schema.as_ref())
            .map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("state_id", result.state_id)?;
        output.set_item("added_nodes", result.added_nodes)?;
        output.set_item("added_edges", result.added_edges)?;
        Ok(output.to_object(py))
    }
    /// Retain a frozen source revision; preparation runs without the GIL.
    #[pyo3(signature=(source, kind="group"))]
    fn retain_group(
        &mut self,
        py: Python<'_>,
        source: &crate::ffi::subgraphs::view::PyGraphView,
        kind: &str,
    ) -> PyResult<usize> {
        let revision = py
            .allow_threads(|| {
                groggy::storage::retained_group::RetainedGroup::new(
                    source.inner.clone(),
                    kind.to_owned(),
                )
            })
            .map_err(graph_error_to_py_err)?;
        self.core_mut()?
            .retain_group_revision(revision)
            .map_err(graph_error_to_py_err)
    }
    /// Publish a meta-node referencing a persisted frozen group.
    #[pyo3(signature=(group, *, effective_at=None))]
    fn create_meta_node(&mut self, group: usize, effective_at: Option<&PyAny>) -> PyResult<usize> {
        let at = effective_at
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        let state = self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .current_state_id();
        self.core_mut()?
            .create_meta_node(group, state, at)
            .map_err(graph_error_to_py_err)
    }
    /// Reconstruct immutable effective-time state from a pinned published head.
    #[pyo3(signature=(effective_at, *, recorded_by=None, legacy_times=None, group_sources=None))]
    fn read_view_at(
        &self,
        py: Python<'_>,
        effective_at: &PyAny,
        recorded_by: Option<&PyAny>,
        legacy_times: Option<&PyDict>,
        group_sources: Option<&PyDict>,
    ) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        let effective_at = crate::ffi::api::graph_batch::timestamp(effective_at)?;
        let recorded_by = recorded_by
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        let legacy = crate::ffi::api::graph_batch::parse_legacy_times(legacy_times)?;
        let sources = crate::ffi::api::graph_batch::parse_group_sources(group_sources)?;
        let options = groggy::state::effective_read::EffectiveReadOptions {
            recorded_by,
            legacy_publications: legacy,
            group_sources: sources,
        };
        let plan = self
            .core()?
            .prepare_effective_read(effective_at, options)
            .map_err(graph_error_to_py_err)?;
        py.allow_threads(move || plan.read_view())
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }

    /// Verify retained memory/schema; reopen verifies physical journal bytes.
    fn check_integrity(&self, py: Python<'_>) -> PyResult<PyObject> {
        let plan = self
            .core()?
            .prepare_integrity_check()
            .map_err(graph_error_to_py_err)?;
        let report = py
            .allow_threads(move || plan.execute())
            .map_err(graph_error_to_py_err)?;
        super::graph::integrity_report_to_py(py, report)
    }

    /// Capture the current accepted durable head.
    fn read_view(&self) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        self.core()?
            .read_view()
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }

    #[pyo3(signature=(view,batch,message,author,*,schema=None))]
    fn apply_view_batch(
        &mut self,
        py: Python,
        view: &crate::ffi::subgraphs::view::PyGraphView,
        batch: &PyAny,
        message: String,
        author: String,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let batch = parse_batch(batch)?;
        let schema = schema.map(parse_schema).transpose()?;
        let result = self
            .core_mut()?
            .apply_view_batch_with_schema(&view.inner, batch, message, author, schema.as_ref())
            .map_err(graph_error_to_py_err)?;
        crate::ffi::api::graph_batch::view_result_to_python(py, result)
    }

    #[pyo3(signature=(rows,expected_state,*,author="upsert",uid_key="external_id",schema=None,effective_at=None))]
    fn publish_upsert_nodes(
        &mut self,
        py: Python,
        rows: &PyAny,
        expected_state: StateId,
        author: &str,
        uid_key: &str,
        schema: Option<&PyDict>,
        effective_at: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let rows = parse_node_upserts(
            self.core()?.graph().map_err(graph_error_to_py_err)?,
            rows,
            uid_key,
        )?;
        let schema = schema.map(parse_schema).transpose()?;
        let at = effective_at.map(timestamp).transpose()?;
        let (ids, state) = self
            .core_mut()?
            .upsert_nodes(rows, expected_state, author.to_owned(), at, schema.as_ref())
            .map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("node_ids", ids)?;
        output.set_item("state_id", state)?;
        Ok(output.to_object(py))
    }
    #[pyo3(signature=(rows,expected_state,*,author="upsert",uid_key="external_id",schema=None,effective_at=None))]
    fn publish_upsert_edges(
        &mut self,
        py: Python,
        rows: &PyAny,
        expected_state: StateId,
        author: &str,
        uid_key: &str,
        schema: Option<&PyDict>,
        effective_at: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let rows = parse_edge_upserts(
            self.core()?.graph().map_err(graph_error_to_py_err)?,
            rows,
            uid_key,
        )?;
        let schema = schema.map(parse_schema).transpose()?;
        let at = effective_at.map(timestamp).transpose()?;
        let (ids, state) = self
            .core_mut()?
            .upsert_edges(rows, expected_state, author.to_owned(), at, schema.as_ref())
            .map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("edge_ids", ids)?;
        output.set_item("state_id", state)?;
        Ok(output.to_object(py))
    }
    fn publication_metadata(&self, py: Python, state_id: StateId) -> PyResult<PyObject> {
        match self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .publication_metadata(state_id)
            .map_err(graph_error_to_py_err)?
        {
            Some(value) => publication_to_python(py, &value),
            None => Ok(py.None()),
        }
    }
    #[pyo3(signature=(state_id,*,owned=false))]
    fn snapshot_at_commit(&self, state_id: StateId, owned: bool) -> PyResult<PyTemporalSnapshot> {
        let graph = self.core()?.graph().map_err(graph_error_to_py_err)?;
        let inner = if owned {
            graph.owned_snapshot_at_commit(state_id)
        } else {
            graph.snapshot_at_commit(state_id)
        }
        .map_err(graph_error_to_py_err)?;
        Ok(PyTemporalSnapshot { inner })
    }
    #[pyo3(signature=(timestamp,*,owned=false))]
    fn snapshot_at_timestamp(&self, timestamp: u64, owned: bool) -> PyResult<PyTemporalSnapshot> {
        let graph = self.core()?.graph().map_err(graph_error_to_py_err)?;
        let inner = if owned {
            graph.owned_snapshot_at_timestamp(timestamp)
        } else {
            graph.snapshot_at_timestamp(timestamp)
        }
        .map_err(graph_error_to_py_err)?;
        Ok(PyTemporalSnapshot { inner })
    }

    #[pyo3(signature=(name,*,expected_state=None))]
    fn prune_branch(&mut self, name: String, expected_state: Option<StateId>) -> PyResult<usize> {
        let expected = expected_state.unwrap_or(
            self.core()?
                .graph()
                .map_err(graph_error_to_py_err)?
                .current_state_id(),
        );
        self.core_mut()?
            .prune_branch(name, expected)
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(*,expected_state=None))]
    fn gc_history(&mut self, expected_state: Option<StateId>) -> PyResult<usize> {
        let expected = expected_state.unwrap_or(
            self.core()?
                .graph()
                .map_err(graph_error_to_py_err)?
                .current_state_id(),
        );
        self.core_mut()?
            .gc_history(expected)
            .map_err(graph_error_to_py_err)
    }

    #[pyo3(signature=(message,author,*,expected_state=None,schema=None))]
    fn checkpoint(
        &mut self,
        message: String,
        author: String,
        expected_state: Option<StateId>,
        schema: Option<&PyDict>,
    ) -> PyResult<StateId> {
        let schema = schema.map(parse_schema).transpose()?;
        let expected = match expected_state {
            Some(state) => state,
            None => self
                .core()?
                .graph()
                .map_err(graph_error_to_py_err)?
                .current_state_id(),
        };
        self.core_mut()?
            .checkpoint_with_schema(expected, message, author, schema.as_ref())
            .map_err(graph_error_to_py_err)
    }
    /// An explicit checkpoint; durable mutations have already been published.
    fn commit(&mut self, message: String, author: String) -> PyResult<StateId> {
        self.checkpoint(message, author, None, None)
    }
    fn checkpoints(&self, py: Python) -> PyResult<PyObject> {
        super::graph_version::checkpoints_to_python(
            py,
            self.core()?
                .graph()
                .map_err(graph_error_to_py_err)?
                .checkpoints(),
        )
    }
    fn commit_history(&self) -> PyResult<Vec<super::graph_version::PyCommit>> {
        Ok(self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .history_records()
            .into_iter()
            .map(super::graph_version::PyCommit::from_core_commit)
            .collect())
    }
    fn configuration(&self, py: Python) -> PyResult<PyObject> {
        let json = serde_json::to_string(
            self.core()?
                .graph()
                .map_err(graph_error_to_py_err)?
                .configuration(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(py
            .import("json")?
            .call_method1("loads", (json,))?
            .to_object(py))
    }
    fn schema(&self, py: Python) -> PyResult<PyObject> {
        if let Some(schema) = self.core()?.schema() {
            let json =
                serde_json::to_string(schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(py
                .import("json")?
                .call_method1("loads", (json,))?
                .to_object(py))
        } else {
            Ok(py.None())
        }
    }
    #[pyo3(signature=(schema,expected_state,expected_revision))]
    fn set_schema(
        &mut self,
        schema: Option<&PyDict>,
        expected_state: StateId,
        expected_revision: u64,
    ) -> PyResult<u64> {
        let schema = schema.map(parse_schema).transpose()?;
        self.core_mut()?
            .set_schema(schema, expected_state, expected_revision)
            .map_err(graph_error_to_py_err)
    }
    fn create_branch(&mut self, name: String, expected_state: StateId) -> PyResult<()> {
        self.core_mut()?
            .create_branch(name, expected_state)
            .map_err(graph_error_to_py_err)
    }
    fn checkout_branch(&mut self, name: String, expected_state: StateId) -> PyResult<()> {
        self.core_mut()?
            .checkout_branch(name, expected_state)
            .map_err(graph_error_to_py_err)
    }
    fn list_branches(&self, py: Python) -> PyResult<PyObject> {
        let rows = PyList::empty(py);
        for branch in self
            .core()?
            .graph()
            .map_err(graph_error_to_py_err)?
            .list_branches()
        {
            let row = PyDict::new(py);
            row.set_item("name", branch.name)?;
            row.set_item("head", branch.head)?;
            row.set_item("is_current", branch.is_current)?;
            rows.append(row)?;
        }
        Ok(rows.to_object(py))
    }
}

fn storage_checkpoint_report_to_py(
    py: Python<'_>,
    report: &groggy::storage::journal::StorageCheckpointReport,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("previous_bytes", report.previous_bytes)?;
    result.set_item("current_bytes", report.current_bytes)?;
    result.set_item("previous_records", report.previous_records)?;
    result.set_item("retained_versions", report.retained_versions)?;
    result.set_item("selected_state", report.selected_state)?;
    if let Some(compaction) = &report.pool_compaction {
        result.set_item(
            "pool_compaction",
            super::graph::pool_compaction_report_to_py(py, compaction)?,
        )?;
    }
    Ok(result.into())
}
