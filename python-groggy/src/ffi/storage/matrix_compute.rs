//! FFI-only ownership/error plumbing. All numerical algorithms remain native.
use groggy::{
    storage::{ComplexEigenResult, GraphMatrix, NumericalMatrixData, SolveReport},
    GraphResult,
};
use pyo3::prelude::*;
pub(crate) trait Output: Sized {
    type Wire: Send;
    fn pack(self) -> GraphResult<Self::Wire>;
    fn unpack(wire: Self::Wire) -> GraphResult<Self>;
}
impl Output for GraphMatrix {
    type Wire = NumericalMatrixData<f64>;
    fn pack(self) -> GraphResult<Self::Wire> {
        self.numerical_data()
    }
    fn unpack(wire: Self::Wire) -> GraphResult<Self> {
        wire.into_matrix()
    }
}
macro_rules! scalar_output { ($($ty:ty),*) => {$(impl Output for $ty {
    type Wire=Self;
    fn pack(self)->GraphResult<Self> { Ok(self) }
    fn unpack(wire:Self)->GraphResult<Self> { Ok(wire) }
})*}; }
scalar_output!(f64, usize, Vec<f64>);
impl<A: Output, B: Output> Output for (A, B) {
    type Wire = (A::Wire, B::Wire);
    fn pack(self) -> GraphResult<Self::Wire> {
        Ok((self.0.pack()?, self.1.pack()?))
    }
    fn unpack(wire: Self::Wire) -> GraphResult<Self> {
        Ok((A::unpack(wire.0)?, B::unpack(wire.1)?))
    }
}
impl<A: Output, B: Output, C: Output> Output for (A, B, C) {
    type Wire = (A::Wire, B::Wire, C::Wire);
    fn pack(self) -> GraphResult<Self::Wire> {
        Ok((self.0.pack()?, self.1.pack()?, self.2.pack()?))
    }
    fn unpack(wire: Self::Wire) -> GraphResult<Self> {
        Ok((A::unpack(wire.0)?, B::unpack(wire.1)?, C::unpack(wire.2)?))
    }
}
impl Output for SolveReport<f64> {
    type Wire = (NumericalMatrixData<f64>, f64, usize);
    fn pack(self) -> GraphResult<Self::Wire> {
        Ok((
            self.solution.pack()?,
            self.backward_error,
            self.accepted_refinements,
        ))
    }
    fn unpack(wire: Self::Wire) -> GraphResult<Self> {
        Ok(Self {
            solution: GraphMatrix::unpack(wire.0)?,
            backward_error: wire.1,
            accepted_refinements: wire.2,
        })
    }
}
impl Output for ComplexEigenResult<f64> {
    type Wire = (
        Vec<f64>,
        Vec<f64>,
        NumericalMatrixData<f64>,
        NumericalMatrixData<f64>,
    );
    fn pack(self) -> GraphResult<Self::Wire> {
        Ok((
            self.eigenvalues_real,
            self.eigenvalues_imag,
            self.eigenvectors_real.pack()?,
            self.eigenvectors_imag.pack()?,
        ))
    }
    fn unpack(wire: Self::Wire) -> GraphResult<Self> {
        Ok(Self {
            eigenvalues_real: wire.0,
            eigenvalues_imag: wire.1,
            eigenvectors_real: GraphMatrix::unpack(wire.2)?,
            eigenvectors_imag: GraphMatrix::unpack(wire.3)?,
        })
    }
}
/// Only Send owned data cross allow_threads. Tracked matrices keep their tape
/// ownership and run on the existing GIL path.
pub(crate) fn run<const N: usize, R: Output>(
    py: Python<'_>,
    inputs: [&GraphMatrix; N],
    operation: impl FnOnce([&GraphMatrix; N]) -> GraphResult<R> + Send,
) -> GraphResult<R> {
    if inputs.iter().any(|matrix| matrix.requires_grad_enabled()) {
        return operation(inputs);
    }
    let snapshots = inputs
        .iter()
        .map(|matrix| matrix.numerical_data())
        .collect::<GraphResult<Vec<_>>>()?;
    let wire = py
        .allow_threads(move || {
            let matrices = snapshots
                .into_iter()
                .map(NumericalMatrixData::into_matrix)
                .collect::<GraphResult<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            operation(std::array::from_fn(|i| &matrices[i]))
                .and_then(Output::pack)
                .map_err(|e| e.to_string())
        })
        .map_err(groggy::GraphError::InvalidInput)?;
    R::unpack(wire)
}
