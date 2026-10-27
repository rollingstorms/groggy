//! Thin immutable coordinate translation. Rust owns all validation semantics.
use crate::ffi::utils::graph_error_to_py_err;
use pyo3::prelude::*;
#[pyclass(name = "WriteCoordinate")]
#[derive(Clone)]
pub struct PyWriteCoordinate {
    pub inner: groggy::WriteCoordinate,
}
#[pymethods]
impl PyWriteCoordinate {
    #[getter]
    fn state_id(&self) -> u64 {
        self.inner.state_id()
    }
    #[getter]
    fn branch(&self) -> &str {
        self.inner.branch()
    }
    fn _advance(&self, state: &PyAny) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.at_state(super::graph_batch::state_id(state)?),
        })
    }
}
pub(super) fn validate(
    graph: &groggy::Graph,
    coordinate: Option<&PyWriteCoordinate>,
) -> PyResult<()> {
    if let Some(coordinate) = coordinate {
        graph
            .validate_write_coordinate(&coordinate.inner)
            .map_err(graph_error_to_py_err)?;
    }
    Ok(())
}
