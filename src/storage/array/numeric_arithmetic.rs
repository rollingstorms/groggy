//! Native numeric conversion and checked vector arithmetic policies.
use super::NumArray;
pub use crate::storage::numeric::scalar::*;
use crate::storage::numeric::NumericType;
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
use crate::{GraphError, GraphResult};
impl NumArray<i64> {
    /// Preserve native unsigned IDs or error when the signed array cannot represent them.
    pub fn from_usize_ids(values: &[usize]) -> GraphResult<Self> {
        values
            .iter()
            .map(|value| {
                i64::try_from(*value)
                    .map_err(|_| invalid("Native ID exceeds signed int64 NumArray range"))
            })
            .collect::<GraphResult<Vec<_>>>()
            .map(Self::new)
    }
}
pub fn divide(a: &[NumericInput], b: &[NumericInput]) -> GraphResult<NumArray<f64>> {
    if a.len() != b.len() {
        return Err(invalid("Vector lengths differ"));
    }
    let values = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| {
            let divisor = b.as_f64();
            if divisor == 0. {
                Err(invalid("Vector division by zero"))
            } else {
                Ok(a.as_f64() / divisor)
            }
        })
        .collect::<GraphResult<Vec<_>>>()?;
    Ok(NumArray::new(values))
}
impl<T: NumericType> NumArray<T> {
    /// Strict numeric attribute conversion with an explicit precision policy.
    pub fn from_attributes(
        values: &[crate::types::AttrValue],
        policy: ConversionPolicy,
    ) -> GraphResult<Self> {
        let (_, values) = attribute_inputs(values)?;
        Self::from_inputs(&values, policy)
    }

    /// Construct with an explicit conversion/rounding policy.
    pub fn from_inputs(values: &[NumericInput], policy: ConversionPolicy) -> GraphResult<Self> {
        Ok(Self::new(
            values
                .iter()
                .copied()
                .map(|v| convert(v, policy))
                .collect::<GraphResult<Vec<_>>>()?,
        ))
    }
    /// Exact representability cast; use an explicit policy to permit rounding.
    pub fn cast<U: NumericType>(&self) -> GraphResult<NumArray<U>> {
        self.cast_with_policy(ConversionPolicy::Exact)
    }
    pub fn cast_with_policy<U: NumericType>(
        &self,
        policy: ConversionPolicy,
    ) -> GraphResult<NumArray<U>> {
        Ok(NumArray::new(
            self.iter()
                .copied()
                .map(|v| convert(from_scalar(v), policy))
                .collect::<GraphResult<Vec<_>>>()?,
        ))
    }
    pub fn checked_binary(&self, other: &Self, operation: NumericBinary) -> GraphResult<Self> {
        if self.len() != other.len() {
            return Err(invalid("Vector lengths differ"));
        }
        let values = self
            .iter()
            .zip(other.iter())
            .map(|(&a, &b)| {
                match operation {
                    NumericBinary::Add => a.checked_add(b),
                    NumericBinary::Subtract => a.checked_sub(b),
                    NumericBinary::Multiply => a.checked_mul(b),
                }
                .ok_or_else(|| invalid("Vector element arithmetic overflow"))
            })
            .collect::<GraphResult<Vec<_>>>()?;
        Ok(Self::new(values))
    }
    pub fn checked_add(&self, other: &Self) -> GraphResult<Self> {
        self.checked_binary(other, NumericBinary::Add)
    }
    pub fn checked_subtract(&self, other: &Self) -> GraphResult<Self> {
        self.checked_binary(other, NumericBinary::Subtract)
    }
    pub fn checked_multiply(&self, other: &Self) -> GraphResult<Self> {
        self.checked_binary(other, NumericBinary::Multiply)
    }
    /// Dtype-preserving scalar operation; scalar conversion must be exact.
    pub fn scalar_binary(
        &self,
        scalar: NumericInput,
        operation: NumericBinary,
        reverse: bool,
    ) -> GraphResult<Self> {
        let policy = if T::DTYPE.is_float() {
            ConversionPolicy::Round
        } else {
            ConversionPolicy::Exact
        };
        let scalar = convert::<T>(scalar, policy)?;
        let values = self
            .iter()
            .map(|&value| {
                let (a, b) = if reverse {
                    (scalar, value)
                } else {
                    (value, scalar)
                };
                match operation {
                    NumericBinary::Add => a.checked_add(b),
                    NumericBinary::Subtract => a.checked_sub(b),
                    NumericBinary::Multiply => a.checked_mul(b),
                }
                .ok_or_else(|| invalid("Vector scalar arithmetic overflow"))
            })
            .collect::<GraphResult<Vec<_>>>()?;
        Ok(Self::new(values))
    }
    /// Accurate typed dot; checked integer subtotals/final output and bool semiring.
    pub fn dot(&self, other: &Self) -> GraphResult<T> {
        if self.len() != other.len() {
            return Err(invalid("Vector dot lengths differ"));
        }
        let mut dot = crate::storage::numeric::checked::Product::<T>::new();
        for (&a, &b) in self.iter().zip(other.iter()) {
            dot.add(a, b).map_err(|e| invalid(&e.to_string()))?;
        }
        dot.total().map_err(|e| invalid(&e.to_string()))
    }
    /// Floating true division; integer operands convert to the documented f64 output.
    pub fn true_divide<U: NumericType>(&self, other: &NumArray<U>) -> GraphResult<NumArray<f64>> {
        divide(&inputs(self.as_slice()), &inputs(other.as_slice()))
    }
    /// Stable Euclidean norm; NaN propagates and output is f64.
    pub fn norm(&self) -> f64 {
        if self.iter().any(|v| v.to_f64().is_nan()) {
            return f64::NAN;
        }
        self.iter().fold(0.0_f64, |norm, v| norm.hypot(v.to_f64()))
    }
    /// Attribute materialization preserves explicit floating widths and checked integers.
    pub fn to_attributes(&self) -> GraphResult<Vec<crate::types::AttrValue>> {
        self.to_attributes_with_policy(ConversionPolicy::Round)
    }
    /// Attribute conversion with explicit floating representability policy.
    pub fn to_attributes_with_policy(
        &self,
        policy: ConversionPolicy,
    ) -> GraphResult<Vec<crate::types::AttrValue>> {
        self.iter()
            .copied()
            .map(|v| attribute_from_scalar(v, policy))
            .collect()
    }
}

/// Scalar true division, returning owned f64 values and rejecting zero divisors.
pub fn divide_scalar<T: NumericType>(
    values: &[T],
    scalar: NumericInput,
    reverse: bool,
) -> GraphResult<NumArray<f64>> {
    let scalar = scalar.as_f64();
    let values = values
        .iter()
        .map(|&v| {
            let (a, b) = if reverse {
                (scalar, v.to_f64())
            } else {
                (v.to_f64(), scalar)
            };
            if b == 0. {
                Err(invalid("Vector division by zero"))
            } else {
                Ok(a / b)
            }
        })
        .collect::<GraphResult<Vec<_>>>()?;
    Ok(NumArray::new(values))
}
/// Exact typed membership, including normal IEEE equality rules.
impl<T: NumericType> NumArray<T> {
    /// First-occurrence unique values; signed zero compares equal and all NaNs coalesce.
    pub fn unique(&self) -> Self {
        #[derive(Hash, PartialEq, Eq)]
        enum Key {
            Integer(i128),
            Float(u64),
        }
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for &value in self.iter() {
            let key = if let Some(value) = value.to_i128() {
                Key::Integer(value)
            } else {
                let value = value.to_f64();
                Key::Float(if value == 0. {
                    0
                } else if value.is_nan() {
                    f64::NAN.to_bits()
                } else {
                    value.to_bits()
                })
            };
            if seen.insert(key) {
                out.push(value);
            }
        }
        Self::new(out)
    }
    /// Transfer an owned vector, copying only if its immutable backing is shared.
    pub fn into_vec(self) -> Vec<T> {
        match std::sync::Arc::try_unwrap(self.base.inner) {
            Ok(values) => values,
            Err(shared) => shared.as_ref().clone(),
        }
    }
}
