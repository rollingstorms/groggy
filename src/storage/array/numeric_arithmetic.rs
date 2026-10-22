//! Native numeric conversion and checked vector arithmetic policies.
use super::NumArray;
use crate::storage::advanced_matrix::{DType, NumericType};
use crate::{GraphError, GraphResult};
#[derive(Clone, Copy, Debug)]
pub enum NumericInput {
    Bool(bool),
    Integer(i128),
    Float(f64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionPolicy {
    Exact,
    Round,
    Lossy,
}
#[derive(Clone, Copy, Debug)]
pub enum NumericBinary {
    Add,
    Subtract,
    Multiply,
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
impl NumericInput {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Bool(v) => {
                if v {
                    1.
                } else {
                    0.
                }
            }
            Self::Integer(v) => v as f64,
            Self::Float(v) => v,
        }
    }
}
pub fn from_scalar<T: NumericType>(value: T) -> NumericInput {
    if T::DTYPE == DType::Bool {
        NumericInput::Bool(value != T::zero())
    } else if let Some(value) = value.to_i128() {
        NumericInput::Integer(value)
    } else {
        NumericInput::Float(value.to_f64())
    }
}
pub fn inputs<T: NumericType>(values: &[T]) -> Vec<NumericInput> {
    values.iter().copied().map(from_scalar).collect()
}
/// Native inference for the legacy numeric attribute bridge (bool/small ints ->
/// int32, larger ints -> int64, any float -> float64). Null is not numeric zero.
pub fn attribute_inputs(
    values: &[crate::types::AttrValue],
) -> GraphResult<(DType, Vec<NumericInput>)> {
    use crate::types::AttrValue;
    let mut floating = false;
    let mut wide = false;
    let inputs = values
        .iter()
        .map(|value| {
            Ok(match value {
                AttrValue::Bool(value) => NumericInput::Bool(*value),
                AttrValue::SmallInt(value) => NumericInput::Integer(i128::from(*value)),
                AttrValue::Int(value) => {
                    wide |= i32::try_from(*value).is_err();
                    NumericInput::Integer(i128::from(*value))
                }
                AttrValue::Float(value) => {
                    floating = true;
                    NumericInput::Float(f64::from(*value))
                }
                _ => return Err(invalid("NumArray can only contain numerical values")),
            })
        })
        .collect::<GraphResult<Vec<_>>>()?;
    Ok((
        if floating {
            DType::Float64
        } else if wide {
            DType::Int64
        } else {
            DType::Int32
        },
        inputs,
    ))
}
/// BaseArray promotion keeps bool-only arrays, widens integers to int64 and
/// floating/mixed arrays to float64. Missing values require explicit handling.
pub fn promoted_attribute_inputs(
    values: &[crate::types::AttrValue],
) -> GraphResult<(DType, Vec<NumericInput>)> {
    let (dtype, inputs) = attribute_inputs(values)?;
    let dtype = if inputs.is_empty() || dtype == DType::Float64 {
        DType::Float64
    } else if inputs
        .iter()
        .all(|value| matches!(value, NumericInput::Bool(_)))
    {
        DType::Bool
    } else {
        DType::Int64
    };
    Ok((dtype, inputs))
}
/// Exact integer positions: fractional/nonfinite/out-of-range indices error.
pub fn integer_indices<T: NumericType>(values: &[T]) -> GraphResult<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| convert(from_scalar(value), ConversionPolicy::Exact))
        .collect()
}
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
pub fn convert<T: NumericType>(input: NumericInput, policy: ConversionPolicy) -> GraphResult<T> {
    let fail = || invalid("Numeric conversion loses precision or exceeds dtype range");
    if T::DTYPE == DType::Bool {
        let value = input.as_f64();
        if policy == ConversionPolicy::Exact && value != 0. && value != 1. {
            return Err(fail());
        }
        return T::from_f64(if value != 0. { 1. } else { 0. }).ok_or_else(fail);
    }
    if T::DTYPE.is_integer() {
        let value = match input {
            NumericInput::Integer(v) => v,
            NumericInput::Bool(v) => i128::from(v),
            NumericInput::Float(v) => {
                if !v.is_finite() {
                    return Err(fail());
                }
                let rounded = match policy {
                    ConversionPolicy::Round => v.round(),
                    ConversionPolicy::Lossy => v.trunc(),
                    ConversionPolicy::Exact => {
                        if v.fract() != 0. {
                            return Err(fail());
                        }
                        v
                    }
                };
                if rounded >= 2f64.powi(127) || rounded < -2f64.powi(127) {
                    return Err(fail());
                }
                rounded as i128
            }
        };
        return T::from_i128(value).ok_or_else(fail);
    }
    let source = input.as_f64();
    let result = T::from_f64(source).ok_or_else(fail)?;
    let output = result.to_f64();
    if source.is_finite() && !output.is_finite() {
        return Err(fail());
    }
    if policy == ConversionPolicy::Exact {
        if let NumericInput::Integer(value) = input {
            if !output.is_finite()
                || output >= 2f64.powi(127)
                || output < -2f64.powi(127)
                || (output as i128) != value
            {
                return Err(fail());
            }
        } else if output != source && !(output.is_nan() && source.is_nan()) {
            return Err(fail());
        }
    }
    Ok(result)
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
        let mut dot = crate::storage::advanced_matrix::checked_kernels::Product::<T>::new();
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
    /// Legacy attribute materialization with checked finite f32/int narrowing.
    pub fn to_attributes(&self) -> GraphResult<Vec<crate::types::AttrValue>> {
        self.to_attributes_with_policy(ConversionPolicy::Round)
    }
    /// Attribute conversion with explicit floating representability policy.
    pub fn to_attributes_with_policy(
        &self,
        policy: ConversionPolicy,
    ) -> GraphResult<Vec<crate::types::AttrValue>> {
        use crate::types::AttrValue;
        inputs(self.as_slice())
            .into_iter()
            .map(|v| match v {
                NumericInput::Bool(v) => Ok(AttrValue::Bool(v)),
                NumericInput::Integer(v) if T::DTYPE == DType::Int32 => i32::try_from(v)
                    .map(AttrValue::SmallInt)
                    .map_err(|_| invalid("Integer attribute range exceeded")),
                NumericInput::Integer(v) => i64::try_from(v)
                    .map(AttrValue::Int)
                    .map_err(|_| invalid("Integer attribute range exceeded")),
                NumericInput::Float(_) => convert::<f32>(v, policy).map(AttrValue::Float),
            })
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
pub fn contains<T: PartialEq>(values: &[T], value: &T) -> bool {
    values.contains(value)
}
/// Scaled accurate sum-of-squares norm without a projection allocation.
pub fn norm<T: NumericType>(values: &[T]) -> f64 {
    let mut scale = 0.0_f64;
    for value in values {
        let value = value.to_f64();
        if value.is_nan() {
            return f64::NAN;
        }
        scale = scale.max(value.abs());
    }
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    let mut squares = crate::storage::matrix::dot_product::Dot::new();
    for value in values {
        let normalized = value.to_f64() / scale;
        squares.add(normalized, normalized);
    }
    squares.total().sqrt() * scale
}

/// Numeric comparison operators; NaN follows IEEE unordered comparison rules.
#[derive(Clone, Copy, Debug)]
pub enum NumericComparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}
pub fn compare_inputs(a: NumericInput, b: NumericInput) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let a = match a {
        NumericInput::Bool(v) => NumericInput::Integer(i128::from(v)),
        v => v,
    };
    let b = match b {
        NumericInput::Bool(v) => NumericInput::Integer(i128::from(v)),
        v => v,
    };
    let int_float = |integer: i128, float: f64| {
        if float.is_nan() {
            return None;
        }
        if float < -2f64.powi(127) {
            return Some(Ordering::Greater);
        }
        if float >= 2f64.powi(127) {
            return Some(Ordering::Less);
        }
        let truncated = float as i128;
        Some(match integer.cmp(&truncated) {
            Ordering::Equal => (truncated as f64).partial_cmp(&float)?,
            order => order,
        })
    };
    match (a, b) {
        (NumericInput::Integer(a), NumericInput::Integer(b)) => Some(a.cmp(&b)),
        (NumericInput::Integer(a), NumericInput::Float(b)) => int_float(a, b),
        (NumericInput::Float(a), NumericInput::Integer(b)) => {
            int_float(b, a).map(Ordering::reverse)
        }
        (a, b) => a.as_f64().partial_cmp(&b.as_f64()),
    }
}
fn comparison(a: NumericInput, b: NumericInput, op: NumericComparison) -> bool {
    use std::cmp::Ordering;
    let order = compare_inputs(a, b);
    match op {
        NumericComparison::Equal => order == Some(Ordering::Equal),
        NumericComparison::NotEqual => order != Some(Ordering::Equal),
        NumericComparison::Less => order == Some(Ordering::Less),
        NumericComparison::LessEqual => matches!(order, Some(Ordering::Less | Ordering::Equal)),
        NumericComparison::Greater => order == Some(Ordering::Greater),
        NumericComparison::GreaterEqual => {
            matches!(order, Some(Ordering::Greater | Ordering::Equal))
        }
    }
}
pub fn compare(
    a: &[NumericInput],
    b: &[NumericInput],
    op: NumericComparison,
) -> GraphResult<Vec<bool>> {
    if a.len() != b.len() {
        return Err(invalid("Vector comparison lengths differ"));
    }
    Ok(a.iter()
        .zip(b)
        .map(|(&a, &b)| comparison(a, b, op))
        .collect())
}
pub fn compare_scalar(
    a: &[NumericInput],
    scalar: NumericInput,
    op: NumericComparison,
) -> Vec<bool> {
    a.iter().map(|&a| comparison(a, scalar, op)).collect()
}
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
