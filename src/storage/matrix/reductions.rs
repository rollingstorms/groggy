//! Shared typed GraphMatrix reductions over checked logical values.
use crate::storage::matrix::checked_kernels::NumericValue as Value;
use crate::storage::{matrix::NumericType, numeric::reductions as k};
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

/// Typed mean with exact integer accumulation and truncation only after division.
pub(super) fn typed_mean<T: NumericType>(values: &[T], count: usize) -> GraphResult<T> {
    if count == 0 {
        return Ok(T::zero());
    }
    if T::DTYPE.is_integer() {
        // Divide before accumulating so the mean of [MAX, MAX] remains valid
        // even for i128. Quotients and remainders retain exact truncation.
        let divisor = count as i128;
        let (mut quotient, mut remainder) = (0_i128, 0_i128);
        for value in values {
            let value = value
                .to_i128()
                .ok_or_else(|| GraphError::InvalidInput("Invalid integer mean input".into()))?;
            quotient = quotient.checked_add(value / divisor).ok_or_else(|| {
                GraphError::InvalidInput("Integer mean accumulator overflow".into())
            })?;
            remainder = remainder.checked_add(value % divisor).ok_or_else(|| {
                GraphError::InvalidInput("Integer mean remainder overflow".into())
            })?;
        }
        quotient = quotient
            .checked_add(remainder / divisor)
            .ok_or_else(|| GraphError::InvalidInput("Integer mean output overflow".into()))?;
        let residual = remainder % divisor;
        // quotient + residual/divisor must truncate toward zero as a whole.
        if quotient > 0 && residual < 0 {
            quotient -= 1;
        }
        if quotient < 0 && residual > 0 {
            quotient += 1;
        }
        return T::from_i128(quotient)
            .ok_or_else(|| GraphError::InvalidInput("Integer mean output overflow".into()));
    }
    statistic_value(extended_mean(values, count))
}
pub(super) fn statistic_value<T: NumericType>(value: f64) -> GraphResult<T> {
    T::from_f64(value).ok_or_else(|| {
        GraphError::InvalidInput("Statistical output cannot be represented in dtype".into())
    })
}
pub(super) fn extended_mean<T: NumericType>(values: &[T], count: usize) -> f64 {
    k::zero_extended_mean(values.iter().copied().map(Value), count).unwrap_or(0.)
}
pub(super) fn extended_std<T: NumericType>(values: &[T], count: usize, sample: bool) -> f64 {
    k::zero_extended_std(values.iter().copied().map(Value), count, sample).unwrap_or(0.)
}
pub(super) fn extended_extremum<T: NumericType>(
    values: &[T],
    count: usize,
    minimum: bool,
) -> GraphResult<T> {
    let best = extremum(values, minimum)?.unwrap_or_else(T::zero);
    if best.to_f64().is_nan() {
        return Ok(best);
    }
    if values.len() < count && ((minimum && best > T::zero()) || (!minimum && best < T::zero())) {
        Ok(T::zero())
    } else {
        Ok(best)
    }
}

pub(super) fn extended_variance<T: NumericType>(values: &[T], count: usize, sample: bool) -> f64 {
    if values.len() == count {
        let values = values.iter().copied().map(Value);
        return if sample {
            k::variance_iter(values)
        } else {
            k::population_variance_iter(values)
        }
        .unwrap_or(0.);
    }
    let std = extended_std(values, count, sample);
    std * std
}
