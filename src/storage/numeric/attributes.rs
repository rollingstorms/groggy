//! Checked attribute statistics: preserve integer hints until the final f32 output.
use super::{
    dot_product::Dot,
    reductions::{self, Scalar},
    scalar::{compare_inputs, convert, ConversionPolicy, NumericInput},
};
use crate::{AttrValue, GraphError, GraphResult};
use std::cmp::Ordering;
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
#[derive(Clone, Copy)]
pub(crate) enum Value {
    Integer(i64),
    Float(f64),
}
impl Value {
    fn parse(value: &AttrValue) -> Option<Self> {
        match value {
            AttrValue::Int(value) => Some(Self::Integer(*value)),
            AttrValue::SmallInt(value) => Some(Self::Integer(i64::from(*value))),
            AttrValue::Float(value) => Some(Self::Float(f64::from(*value))),
            _ => None,
        }
    }
    fn input(self) -> NumericInput {
        match self {
            Self::Integer(v) => NumericInput::Integer(i128::from(v)),
            Self::Float(v) => NumericInput::Float(v),
        }
    }
    fn parts(self) -> (f64, f64) {
        match self {
            Self::Integer(v) => {
                let hi = v as f64;
                (hi, (i128::from(v) - hi as i128) as f64)
            }
            Self::Float(v) => (v, 0.),
        }
    }
}
impl Scalar for Value {
    fn as_f64(self) -> f64 {
        match self {
            Self::Integer(v) => v as f64,
            Self::Float(v) => v,
        }
    }
    fn integer(self) -> Option<i128> {
        match self {
            Self::Integer(v) => Some(i128::from(v)),
            Self::Float(_) => None,
        }
    }
    fn difference(self, origin: Self) -> f64 {
        if let (Self::Integer(a), Self::Integer(b)) = (self, origin) {
            return (i128::from(a) - i128::from(b)) as f64;
        }
        let (hi, lo) = self.parts();
        let (origin_hi, origin_lo) = origin.parts();
        (hi - origin_hi) + (lo - origin_lo)
    }
    fn numeric_cmp(self, other: Self) -> Ordering {
        compare_inputs(self.input(), other.input()).unwrap_or(Ordering::Equal)
    }
}
pub(crate) fn values(input: &[AttrValue]) -> GraphResult<Vec<Value>> {
    let mut output = Vec::new();
    for value in input.iter().filter_map(Value::parse) {
        if !value.as_f64().is_finite() {
            return Err(invalid(
                "Attribute statistics require finite numeric values",
            ));
        }
        output.push(value);
    }
    Ok(output)
}
pub(crate) fn float_output(value: f64) -> GraphResult<AttrValue> {
    if !value.is_finite() {
        return Err(invalid("Attribute statistic exceeds finite working range"));
    }
    convert::<f32>(NumericInput::Float(value), ConversionPolicy::Round).map(AttrValue::Float)
}
/// Numeric quantile, skipping null/nonnumeric cells; output retains attribute f32 storage.
pub fn quantile(input: &[AttrValue], q: f64, interpolation: &str) -> GraphResult<AttrValue> {
    if !(0.0..=1.0).contains(&q) {
        return Err(invalid("Quantile must be finite and between zero and one"));
    }
    if !["linear", "lower", "higher", "midpoint", "nearest"].contains(&interpolation) {
        return Err(invalid("Invalid quantile interpolation method"));
    }
    let mut values = values(input)?;
    if values.is_empty() {
        return Ok(AttrValue::Null);
    }
    values.sort_by(|a, b| a.numeric_cmp(*b));
    let rank = q * (values.len() - 1) as f64;
    let low = (rank.floor() as usize).min(values.len() - 1);
    let high = (rank.ceil() as usize).min(values.len() - 1);
    let result = match interpolation {
        "lower" => values[low].as_f64(),
        "higher" => values[high].as_f64(),
        "nearest" => values[if rank - (low as f64) < 0.5 { low } else { high }].as_f64(),
        _ => {
            let weight = if interpolation == "midpoint" {
                0.5
            } else {
                rank - low as f64
            };
            let mut sum = Dot::new();
            for (value, weight) in [(values[low], 1. - weight), (values[high], weight)] {
                let (hi, lo) = value.parts();
                sum.add(hi, weight);
                sum.add(lo, weight);
            }
            sum.total()
        }
    };
    float_output(result)
}
fn pairs(x: &[AttrValue], y: &[AttrValue]) -> GraphResult<(Vec<Value>, Vec<Value>)> {
    if x.len() != y.len() {
        return Err(invalid("Paired attribute arrays must have equal lengths"));
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (x, y) in x.iter().zip(y) {
        if let (Some(x), Some(y)) = (Value::parse(x), Value::parse(y)) {
            if !x.as_f64().is_finite() || !y.as_f64().is_finite() {
                return Err(invalid(
                    "Paired attribute statistics require finite numeric values",
                ));
            }
            left.push(x);
            right.push(y);
        }
    }
    Ok((left, right))
}
fn ranks(values: &[Value]) -> Vec<f64> {
    let mut ordered: Vec<_> = values.iter().copied().enumerate().collect();
    ordered.sort_by(|a, b| a.1.numeric_cmp(b.1));
    let mut result = vec![0.; values.len()];
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[start].1.numeric_cmp(ordered[end].1) == Ordering::Equal
        {
            end += 1;
        }
        let rank = start as f64 / 2. + (end - 1) as f64 / 2. + 1.;
        for &(position, _) in &ordered[start..end] {
            result[position] = rank;
        }
        start = end;
    }
    result
}
/// Pearson/Spearman or legacy Kendall tau-a; pairwise missing cells are omitted.
/// Fewer than two numeric pairs gives null; constant Pearson/Spearman gives zero.
pub fn correlation(x: &[AttrValue], y: &[AttrValue], method: &str) -> GraphResult<AttrValue> {
    if !["pearson", "spearman", "kendall"].contains(&method) {
        return Err(invalid("Invalid correlation method"));
    }
    let (x, y) = pairs(x, y)?;
    if x.len() < 2 {
        return Ok(AttrValue::Null);
    }
    let result = match method {
        "pearson" => reductions::correlate(&x, &y).unwrap_or(0.),
        "spearman" => reductions::correlate(&ranks(&x), &ranks(&y)).unwrap_or(0.),
        _ => {
            let mut concordant = 0u128;
            let mut discordant = 0u128;
            for i in 0..x.len() {
                for j in i + 1..x.len() {
                    let a = x[i].numeric_cmp(x[j]);
                    let b = y[i].numeric_cmp(y[j]);
                    if a != Ordering::Equal && b != Ordering::Equal {
                        if a == b {
                            concordant += 1;
                        } else {
                            discordant += 1;
                        }
                    }
                }
            }
            let n = x.len() as u128;
            let count = n * (n - 1) / 2;
            (concordant as f64 - discordant as f64) / count as f64
        }
    };
    float_output(result)
}
fn centered(values: &[Value]) -> Vec<f64> {
    let origin = values[0];
    let mut differences: Vec<_> = values
        .iter()
        .map(|&value| value.difference(origin))
        .collect();
    let mean = reductions::mean(&differences).unwrap_or(0.);
    for value in &mut differences {
        *value -= mean;
    }
    differences
}
/// Center in the original integer scale before accumulating covariance products.
pub fn covariance(x: &[AttrValue], y: &[AttrValue], ddof: i32) -> GraphResult<AttrValue> {
    if ddof < 0 {
        return Err(invalid("Covariance ddof must be nonnegative"));
    }
    let (x, y) = pairs(x, y)?;
    if x.len() <= ddof as usize {
        return Ok(AttrValue::Null);
    }
    let denominator = (x.len() - ddof as usize) as f64;
    let x = centered(&x);
    let y = centered(&y);
    let mut product = Dot::new();
    for (&x, &y) in x.iter().zip(&y) {
        product.add(x, y);
    }
    float_output(product.total() / denominator)
}
