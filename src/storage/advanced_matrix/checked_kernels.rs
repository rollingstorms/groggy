//! Typed checked reference arithmetic shared by matrix and vector layers.
use super::{
    unified_matrix::{MatrixError, MatrixResult},
    NumericType,
};
use crate::storage::matrix::dot_product::Dot;
pub(crate) struct Product<T: NumericType> {
    float: Dot,
    integer: i128,
    other: T,
}
impl<T: NumericType> Product<T> {
    pub(crate) fn new() -> Self {
        Self {
            float: Dot::new(),
            integer: 0,
            other: T::zero(),
        }
    }
    pub(crate) fn add(&mut self, a: T, b: T) -> MatrixResult<()> {
        if T::DTYPE.is_float() {
            self.float.add(a.to_f64(), b.to_f64());
        } else if T::DTYPE.is_integer() {
            self.integer = a
                .to_i128()
                .zip(b.to_i128())
                .and_then(|(a, b)| a.checked_mul(b))
                .and_then(|v| self.integer.checked_add(v))
                .ok_or_else(|| {
                    MatrixError::InvalidInput("Integer dot accumulator overflow".into())
                })?;
        } else {
            self.other = a
                .checked_mul(b)
                .and_then(|v| self.other.checked_add(v))
                .ok_or_else(|| MatrixError::InvalidInput("Dot arithmetic overflow".into()))?;
        }
        Ok(())
    }
    pub(crate) fn total(self) -> MatrixResult<T> {
        if T::DTYPE.is_float() {
            T::from_f64(self.float.total())
        } else if T::DTYPE.is_integer() {
            T::from_i128(self.integer)
        } else {
            Some(self.other)
        }
        .ok_or_else(|| MatrixError::InvalidInput("Dot output exceeds dtype range".into()))
    }
}
pub(crate) fn sum<T: NumericType>(values: &[T]) -> MatrixResult<T::Accumulator> {
    let mut product = Product::<T::Accumulator>::new();
    for &value in values {
        let converted = if T::DTYPE.is_integer() {
            value.to_i128().and_then(T::Accumulator::from_i128)
        } else {
            T::Accumulator::from_f64(value.to_f64())
        }
        .ok_or_else(|| {
            MatrixError::InvalidInput("Reduction accumulator conversion failed".into())
        })?;
        product.add(converted, T::Accumulator::one())?;
    }
    product.total()
}
pub(crate) fn extrema<T: NumericType>(values: &[T], minimum: bool) -> MatrixResult<T> {
    let mut best = values
        .first()
        .copied()
        .ok_or_else(|| MatrixError::InvalidInput("Empty matrix extremum".into()))?;
    for &value in values {
        if value.to_f64().is_nan() {
            return Ok(value);
        }
        let order = value
            .partial_cmp(&best)
            .ok_or_else(|| MatrixError::InvalidInput("Unordered matrix value".into()))?;
        if (minimum && order.is_lt()) || (!minimum && order.is_gt()) {
            best = value;
        }
    }
    Ok(best)
}
