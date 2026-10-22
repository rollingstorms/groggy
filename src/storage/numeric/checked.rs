//! Shared typed checked arithmetic; no matrix or array ownership dependencies.
use super::{dot_product::Dot, NumericType};
#[derive(Debug, Clone, Copy)]
pub(crate) struct NumericError(pub &'static str);
impl std::fmt::Display for NumericError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for NumericError {}
pub(crate) type NumericResult<T> = Result<T, NumericError>;
#[derive(Clone, Copy)]
pub(crate) struct NumericValue<T: NumericType>(pub T);
impl<T: NumericType> super::reductions::Scalar for NumericValue<T> {
    fn as_f64(self) -> f64 {
        self.0.to_f64()
    }
    fn integer(self) -> Option<i128> {
        self.0.to_i128()
    }
    fn numeric_cmp(self, other: Self) -> std::cmp::Ordering {
        if let (Some(a), Some(b)) = (self.0.to_i128(), other.0.to_i128()) {
            a.cmp(&b)
        } else {
            let (a, b) = (self.0.to_f64(), other.0.to_f64());
            if a == b {
                std::cmp::Ordering::Equal
            } else {
                a.total_cmp(&b)
            }
        }
    }
}
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
    pub(crate) fn add(&mut self, a: T, b: T) -> NumericResult<()> {
        if T::DTYPE.is_float() {
            self.float.add(a.to_f64(), b.to_f64());
        } else if T::DTYPE.is_integer() {
            self.integer = a
                .to_i128()
                .zip(b.to_i128())
                .and_then(|(a, b)| a.checked_mul(b))
                .and_then(|v| self.integer.checked_add(v))
                .ok_or(NumericError("Integer dot accumulator overflow"))?;
        } else {
            self.other = a
                .checked_mul(b)
                .and_then(|v| self.other.checked_add(v))
                .ok_or(NumericError("Dot arithmetic overflow"))?;
        }
        Ok(())
    }
    pub(crate) fn total(self) -> NumericResult<T> {
        if T::DTYPE.is_float() {
            T::from_f64(self.float.total())
        } else if T::DTYPE.is_integer() {
            T::from_i128(self.integer)
        } else {
            Some(self.other)
        }
        .ok_or(NumericError("Dot output exceeds dtype range"))
    }
}
pub(crate) fn sum<T: NumericType>(values: &[T]) -> NumericResult<T::Accumulator> {
    let mut product = Product::<T::Accumulator>::new();
    for &value in values {
        let converted = if T::DTYPE.is_integer() {
            value.to_i128().and_then(T::Accumulator::from_i128)
        } else {
            T::Accumulator::from_f64(value.to_f64())
        }
        .ok_or(NumericError("Reduction accumulator conversion failed"))?;
        product.add(converted, T::Accumulator::one())?;
    }
    product.total()
}
pub(crate) fn extrema<T: NumericType>(values: &[T], minimum: bool) -> NumericResult<T> {
    let mut best = values
        .first()
        .copied()
        .ok_or(NumericError("Empty matrix extremum"))?;
    for &value in values {
        if value.to_f64().is_nan() {
            return Ok(value);
        }
        let order = value
            .partial_cmp(&best)
            .ok_or(NumericError("Unordered matrix value"))?;
        if (minimum && order.is_lt()) || (!minimum && order.is_gt()) {
            best = value;
        }
    }
    Ok(best)
}
