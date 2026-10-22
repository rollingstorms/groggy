//! Shared typed GraphMatrix reductions over checked logical values.
use crate::storage::advanced_matrix::checked_kernels::NumericValue as Value;
use crate::storage::{advanced_matrix::NumericType, array::numeric_reductions as k};
use crate::{GraphError, GraphResult};
pub(super) fn sum<T: NumericType>(values: &[T]) -> GraphResult<T> {
    if T::DTYPE.is_float() {
        let mut dot = super::dot_product::Dot::new();
        for &value in values {
            dot.add(value.to_f64(), 1.);
        }
        T::from_f64(dot.total())
            .ok_or_else(|| GraphError::InvalidInput("Cannot represent matrix sum".into()))
    } else if T::DTYPE.is_integer() {
        let mut total = 0_i128;
        for &value in values {
            total = value
                .to_i128()
                .and_then(|v| total.checked_add(v))
                .ok_or_else(|| {
                    GraphError::InvalidInput("Integer matrix sum accumulator overflow".into())
                })?;
        }
        T::from_i128(total)
            .ok_or_else(|| GraphError::InvalidInput("Integer matrix sum output overflow".into()))
    } else {
        values.iter().try_fold(T::zero(), |sum, &v| {
            sum.checked_add(v)
                .ok_or_else(|| GraphError::InvalidInput("Matrix sum overflow".into()))
        })
    }
}
pub(super) fn mean<T: NumericType>(values: &[T]) -> f64 {
    k::mean_iter(values.iter().copied().map(Value)).unwrap_or(0.)
}
pub(super) fn variance<T: NumericType>(values: &[T]) -> f64 {
    k::variance_iter(values.iter().copied().map(Value)).unwrap_or(0.)
}
pub(super) fn std<T: NumericType>(values: &[T]) -> f64 {
    k::std_iter(values.iter().copied().map(Value)).unwrap_or(0.)
}
pub(super) fn population_variance<T: NumericType>(values: &[T]) -> f64 {
    k::population_variance_iter(values.iter().copied().map(Value)).unwrap_or(0.)
}
pub(super) fn extremum<T: NumericType>(values: &[T], minimum: bool) -> GraphResult<Option<T>> {
    let mut best = None;
    for &value in values {
        if value.to_f64().is_nan() {
            return Ok(Some(value));
        }
        if let Some(previous) = best {
            let order = value.partial_cmp(&previous).ok_or_else(|| {
                GraphError::InvalidInput("Unordered matrix reduction values".into())
            })?;
            if (minimum && order.is_lt()) || (!minimum && order.is_gt()) {
                best = Some(value);
            }
        } else {
            best = Some(value);
        }
    }
    Ok(best)
}
