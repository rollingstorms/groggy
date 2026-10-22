//! Matrix error adapters over the shared checked numeric kernels.
use super::{
    unified_matrix::{MatrixError, MatrixResult},
    NumericType,
};
use crate::storage::numeric::checked;
pub(crate) use checked::NumericValue;
impl From<checked::NumericError> for MatrixError {
    fn from(error: checked::NumericError) -> Self {
        Self::InvalidInput(error.0.into())
    }
}
pub(crate) struct Product<T: NumericType>(checked::Product<T>);
impl<T: NumericType> Product<T> {
    pub(crate) fn new() -> Self {
        Self(checked::Product::new())
    }
    pub(crate) fn add(&mut self, a: T, b: T) -> MatrixResult<()> {
        self.0.add(a, b).map_err(Into::into)
    }
    pub(crate) fn total(self) -> MatrixResult<T> {
        self.0.total().map_err(Into::into)
    }
}
pub(crate) fn sum<T: NumericType>(values: &[T]) -> MatrixResult<T::Accumulator> {
    checked::sum(values).map_err(Into::into)
}
pub(crate) fn extrema<T: NumericType>(values: &[T], minimum: bool) -> MatrixResult<T> {
    checked::extrema(values, minimum).map_err(Into::into)
}
