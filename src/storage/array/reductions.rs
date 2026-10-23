//! Checked reductions for logical AttrValue columns, shared by both BaseArray APIs.
use crate::{AttrValue, GraphError, GraphResult};
use std::cmp::Ordering;

#[derive(Clone, Copy)]
enum Number {
    Integer(i64),
    Float(f64),
}
impl Number {
    fn from_attr(value: &AttrValue) -> Option<Self> {
        match value {
            AttrValue::Int(v) => Some(Self::Integer(*v)),
            AttrValue::SmallInt(v) => Some(Self::Integer(i64::from(*v))),
            AttrValue::Float(v) => Some(Self::Float(f64::from(*v))),
            AttrValue::Float64(v) => Some(Self::Float(*v)),
            _ => None,
        }
    }
    /// Two-part representation retains integer bits lost by conversion to f64.
    fn parts(self) -> (f64, f64) {
        match self {
            Self::Integer(v) => {
                let hi = v as f64;
                (hi, (i128::from(v) - hi as i128) as f64)
            }
            Self::Float(v) => (v, 0.0),
        }
    }
    fn offset(self, origin: Self) -> f64 {
        if let (Self::Integer(a), Self::Integer(b)) = (self, origin) {
            return (i128::from(a) - i128::from(b)) as f64;
        }
        let (hi, lo) = self.parts();
        let (origin_hi, origin_lo) = origin.parts();
        (hi - origin_hi) + (lo - origin_lo)
    }
    fn cmp(self, other: Self) -> Ordering {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => a.cmp(&b),
            (Self::Float(a), Self::Float(b)) => {
                // Signed zeros must compare equal across floats and integer zero.
                if a == b {
                    Ordering::Equal
                } else {
                    a.total_cmp(&b)
                }
            }
            (Self::Integer(a), Self::Float(b)) => compare_int_float(a, b),
            (Self::Float(a), Self::Integer(b)) => compare_int_float(b, a).reverse(),
        }
    }
}
impl crate::storage::numeric::reductions::Scalar for Number {
    fn as_f64(self) -> f64 {
        self.parts().0
    }
    fn integer(self) -> Option<i128> {
        match self {
            Self::Integer(v) => Some(i128::from(v)),
            _ => None,
        }
    }
    fn difference(self, origin: Self) -> f64 {
        self.offset(origin)
    }
    fn numeric_cmp(self, other: Self) -> Ordering {
        self.cmp(other)
    }
}

#[derive(Clone)]
struct ValidatedNumbers<I> {
    values: I,
    remaining: usize,
}
impl<'a, I: Iterator<Item = &'a AttrValue>> Iterator for ValidatedNumbers<I> {
    type Item = Number;
    fn next(&mut self) -> Option<Number> {
        let value = self.values.find_map(Number::from_attr);
        if value.is_some() {
            self.remaining -= 1;
        } else {
            self.remaining = 0;
        }
        value
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl<'a, I: Iterator<Item = &'a AttrValue>> ExactSizeIterator for ValidatedNumbers<I> {}

fn compare_int_float(integer: i64, float: f64) -> Ordering {
    if float < i64::MIN as f64 {
        return Ordering::Greater;
    }
    // i64::MAX rounds up to 2^63 in f64, which is greater than every i64.
    if float >= -(i64::MIN as f64) {
        return Ordering::Less;
    }
    let truncated = float as i64;
    match integer.cmp(&truncated) {
        Ordering::Equal => {
            if (truncated as f64) < float {
                Ordering::Less
            } else if (truncated as f64) > float {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        order => order,
    }
}

#[derive(Clone, Copy, Default)]
struct Compensated {
    sum: f64,
    correction: f64,
}
impl Compensated {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }
    fn total(self) -> f64 {
        self.sum + self.correction
    }
    fn add_integer(&mut self, integer: i128) {
        let hi = integer as f64;
        self.add(hi);
        self.add((integer - hi as i128) as f64);
    }
}

struct Input<I> {
    values: I,
    count: usize,
    origin: Number,
    integer_sum: i128,
    floats: Compensated,
    has_float: bool,
}
impl<'a, I: Iterator<Item = &'a AttrValue> + Clone> Input<I> {
    fn new(values: I, operation: &str) -> GraphResult<Self> {
        let mut origin = None;
        let mut count = 0;
        let mut integer_sum = 0_i128;
        let mut floats = Compensated::default();
        let mut has_float = false;
        for (index, value) in values.clone().enumerate() {
            if matches!(value, AttrValue::Null) {
                continue;
            }
            let number = Number::from_attr(value).ok_or_else(|| {
                invalid(format!(
                    "Cannot compute {operation} of non-numeric value at index {index}: {value:?}",
                ))
            })?;
            match number {
                Number::Integer(v) => {
                    integer_sum = integer_sum
                        .checked_add(i128::from(v))
                        .ok_or_else(|| invalid("Integer reduction accumulator overflow"))?;
                }
                Number::Float(v) => {
                    if !v.is_finite() {
                        return Err(invalid(format!(
                            "Cannot compute {operation} of nonfinite value at index {index}",
                        )));
                    }
                    floats.add(v);
                    has_float = true;
                }
            }
            if origin.is_none() {
                origin = Some(number);
            }
            count += 1;
        }
        let origin = origin.ok_or_else(|| {
            invalid(format!(
                "Cannot compute {operation} of empty array (no non-null values)"
            ))
        })?;
        Ok(Self {
            values,
            count,
            origin,
            integer_sum,
            floats,
            has_float,
        })
    }
    fn numbers(&self) -> impl ExactSizeIterator<Item = Number> + Clone + '_ {
        ValidatedNumbers {
            values: self.values.clone(),
            remaining: self.count,
        }
    }
    #[cold]
    fn exact_divided(&self, divisor: usize) -> f64 {
        let mut total = crate::storage::numeric::dot_product::Dot::new();
        for value in self.numbers() {
            let (hi, lo) = value.parts();
            total.add(hi, 1.0);
            total.add(lo, 1.0);
        }
        total.total_divided_by(divisor)
    }
    fn total(&self) -> f64 {
        let mut total = self.floats;
        total.add_integer(self.integer_sum);
        total.total()
    }
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn checked(value: f64, operation: &str) -> GraphResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{operation} result is not finite")))
    }
}

pub(super) fn sum(values: &[AttrValue]) -> GraphResult<AttrValue> {
    sum_iter(values.iter())
}
pub(crate) fn sum_iter<'a>(
    values: impl Iterator<Item = &'a AttrValue> + Clone,
) -> GraphResult<AttrValue> {
    let input = Input::new(values, "sum")?;
    if !input.has_float {
        let value = i64::try_from(input.integer_sum)
            .map_err(|_| invalid("Integer sum exceeds i64 output range"))?;
        return Ok(AttrValue::Int(value));
    }
    let value = input.total();
    let value = if value.is_finite() {
        value
    } else {
        input.exact_divided(1)
    };
    if !value.is_finite() {
        return Err(invalid("Float sum exceeds finite f64 output range"));
    }
    Ok(AttrValue::Float64(value))
}
pub(super) fn mean(values: &[AttrValue]) -> GraphResult<f64> {
    mean_iter(values.iter())
}
pub(crate) fn mean_iter<'a>(
    values: impl Iterator<Item = &'a AttrValue> + Clone,
) -> GraphResult<f64> {
    let input = Input::new(values, "mean")?;
    let value = input.total() / input.count as f64;
    checked(
        if value.is_finite() {
            value
        } else {
            input.exact_divided(input.count)
        },
        "Mean",
    )
}
pub(super) fn variance(values: &[AttrValue]) -> GraphResult<f64> {
    variance_iter(values.iter())
}
pub(crate) fn variance_iter<'a>(
    values: impl Iterator<Item = &'a AttrValue> + Clone,
) -> GraphResult<f64> {
    let input = Input::new(values, "variance")?;
    if input.count < 2 {
        return Err(invalid(
            "Cannot compute variance with less than 2 non-null values",
        ));
    }
    // Center exact integer differences before float conversion, then use a
    // compensated two-pass sample variance. This avoids high-offset collapse.
    let mut centered = Compensated::default();
    for number in input.numbers() {
        centered.add(number.offset(input.origin));
    }
    let mean_offset = centered.total() / input.count as f64;
    let mut squared = Compensated::default();
    for number in input.numbers() {
        let difference = number.offset(input.origin) - mean_offset;
        squared.add(difference * difference);
    }
    checked(squared.total() / (input.count - 1) as f64, "Variance")
}
pub(super) fn std(values: &[AttrValue]) -> GraphResult<f64> {
    std_iter(values.iter())
}
pub(crate) fn std_iter<'a>(
    values: impl Iterator<Item = &'a AttrValue> + Clone,
) -> GraphResult<f64> {
    let input = Input::new(values, "standard deviation")?;
    if input.count < 2 {
        return Err(invalid(
            "Cannot compute variance with less than 2 non-null values",
        ));
    }
    let value = crate::storage::numeric::reductions::std_iter(input.numbers())
        .ok_or_else(|| invalid("Cannot compute standard deviation of empty array"))?;
    checked(value, "Standard deviation")
}
pub(super) fn median(values: &[AttrValue]) -> GraphResult<f64> {
    let input = Input::new(values.iter(), "median")?;
    let mut numbers: Vec<_> = input.numbers().collect();
    numbers.sort_by(|a, b| a.cmp(*b));
    let middle = numbers.len() / 2;
    let number = numbers[middle];
    if numbers.len().is_multiple_of(2) {
        let before = numbers[middle - 1];
        if let (Number::Integer(a), Number::Integer(b)) = (before, number) {
            return Ok((i128::from(a) + i128::from(b)) as f64 / 2.0);
        }
        if before.parts().0 == 0.0 && number.parts().0 == 0.0 {
            return Ok((before.parts().0 + number.parts().0) / 2.0);
        }
        let mut total = Compensated::default();
        for item in [before, number] {
            let (hi, lo) = item.parts();
            total.add(hi);
            total.add(lo);
        }
        checked(total.total() / 2.0, "Median")
    } else {
        match number {
            Number::Integer(v) => Ok(v as f64),
            Number::Float(v) => Ok(v),
        }
    }
}

/// Select an original numeric scalar using exact mixed ordering; ties keep first.
pub(super) fn numeric_extremum(values: &[AttrValue], minimum: bool) -> GraphResult<AttrValue> {
    extremum(values, minimum, false)
}
/// Preserve legacy text/boolean extrema alongside the shared numeric domain.
pub(super) fn legacy_extremum(values: &[AttrValue], minimum: bool) -> GraphResult<AttrValue> {
    extremum(values, minimum, true)
}
fn extremum(values: &[AttrValue], minimum: bool, legacy: bool) -> GraphResult<AttrValue> {
    let operation = if minimum { "min" } else { "max" };
    let mut selected: Option<&AttrValue> = None;
    for (index, value) in values.iter().enumerate() {
        if matches!(value, AttrValue::Null) {
            continue;
        }
        match Number::from_attr(value) {
            Some(Number::Float(v)) if !v.is_finite() => return Err(invalid(format!(
                "Cannot compute {operation} of nonfinite value at index {index}",
            ))),
            Some(_) => {},
            None if legacy && matches!(value, AttrValue::Text(_) | AttrValue::CompactText(_) | AttrValue::Bool(_)) => {},
            None => return Err(invalid(format!("Cannot compute {operation} of non-numeric or unsupported value at index {index}: {value:?}"))),
        }
        let Some(previous) = selected else {
            selected = Some(value);
            continue;
        };
        let order = match (Number::from_attr(value), Number::from_attr(previous)) {
            (Some(a), Some(b)) => a.cmp(b),
            _ => match (value, previous) {
                (AttrValue::Text(a), AttrValue::Text(b)) => a.cmp(b),
                (AttrValue::CompactText(a), AttrValue::CompactText(b)) => {
                    a.as_str().cmp(b.as_str())
                }
                (AttrValue::Text(a), AttrValue::CompactText(b)) => a.as_str().cmp(b.as_str()),
                (AttrValue::CompactText(a), AttrValue::Text(b)) => a.as_str().cmp(b.as_str()),
                (AttrValue::Bool(a), AttrValue::Bool(b)) => a.cmp(b),
                _ => {
                    return Err(invalid(format!(
                    "Cannot compare incompatible types for {operation}: {value:?} and {previous:?}"
                )))
                }
            },
        };
        if (minimum && order == Ordering::Less) || (!minimum && order == Ordering::Greater) {
            selected = Some(value);
        }
    }
    selected.cloned().ok_or_else(|| {
        invalid(format!(
            "Cannot compute {operation} of empty array (no non-null values)"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn median_numeric_order_is_transitive_across_integer_limits_and_signed_zero() {
        let numbers = [
            Number::Integer(i64::MIN),
            Number::Integer(i64::MIN + 1),
            Number::Integer(-1),
            Number::Integer(0),
            Number::Integer(1),
            Number::Integer(i64::MAX),
            Number::Float(-f64::from(f32::MAX)),
            Number::Float(-(2.0_f64).powi(63)),
            Number::Float(-0.5),
            Number::Float(-0.0),
            Number::Float(0.0),
            Number::Float(0.5),
            Number::Float((2.0_f64).powi(63)),
            Number::Float(f64::from(f32::MAX)),
        ];
        for &a in &numbers {
            for &b in &numbers {
                for &c in &numbers {
                    assert_eq!(a.cmp(b), b.cmp(a).reverse());
                    if a.cmp(b) != Ordering::Greater && b.cmp(c) != Ordering::Greater {
                        assert_ne!(a.cmp(c), Ordering::Greater);
                    }
                }
            }
        }
    }
}
