//! Checked attribute windows and cumulative statistics with linear expanding work.
use super::{
    attributes::{float_output, Value},
    dot_product::Dot,
    reductions::Scalar,
};
use crate::{AttrValue, GraphError, GraphResult};
use std::cmp::Ordering;
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn validate(operation: &str) -> GraphResult<()> {
    if ["sum", "mean", "min", "max", "std"].contains(&operation) {
        Ok(())
    } else {
        Err(invalid("Unsupported attribute window operation"))
    }
}
fn numeric(value: &AttrValue) -> GraphResult<Option<Value>> {
    let value = Value::parse(value);
    if value.is_some_and(|v| !v.as_f64().is_finite()) {
        return Err(invalid("Attribute windows require finite numeric values"));
    }
    Ok(value)
}
struct Running {
    count: usize,
    sum: Dot,
    origin: Option<Value>,
    mean_offset: f64,
    moment: f64,
    wide_moment: Option<Box<Dot>>,
    min: Option<Value>,
    max: Option<Value>,
}
impl Running {
    fn new() -> Self {
        Self {
            count: 0,
            sum: Dot::new(),
            origin: None,
            mean_offset: 0.,
            moment: 0.,
            wide_moment: None,
            min: None,
            max: None,
        }
    }
    fn add(&mut self, value: &AttrValue) -> GraphResult<()> {
        let Some(value) = numeric(value)? else {
            return Ok(());
        };
        let (hi, lo) = value.parts();
        self.sum.add(hi, 1.);
        self.sum.add(lo, 1.);
        let origin = *self.origin.get_or_insert(value);
        let offset = value.difference(origin);
        self.count += 1;
        let delta = offset - self.mean_offset;
        self.mean_offset += delta / self.count as f64;
        let residual = offset - self.mean_offset;
        if let Some(moment) = self.wide_moment.as_mut() {
            moment.add(delta, residual);
        } else {
            let product = delta * residual;
            let next = self.moment + product;
            if next.is_finite() {
                self.moment = next;
            } else {
                let mut moment = Box::new(Dot::new());
                moment.add(self.moment, 1.0);
                moment.add(delta, residual);
                self.wide_moment = Some(moment);
            }
        }
        if self
            .min
            .is_none_or(|v| value.numeric_cmp(v) == Ordering::Less)
        {
            self.min = Some(value)
        }
        if self
            .max
            .is_none_or(|v| value.numeric_cmp(v) == Ordering::Greater)
        {
            self.max = Some(value)
        }
        Ok(())
    }
    fn output(&self, operation: &str, empty_sum: bool) -> GraphResult<AttrValue> {
        if self.count == 0 {
            return Ok(if empty_sum && operation == "sum" {
                AttrValue::Float64(0.)
            } else {
                AttrValue::Null
            });
        }
        let value = match operation {
            "sum" => self.sum.clone().total(),
            "mean" => {
                let total = self.sum.clone().total();
                if total.is_finite() {
                    total / self.count as f64
                } else {
                    self.sum.clone().total_divided_by(self.count)
                }
            }
            "min" => self.min.map_or(0., Scalar::as_f64),
            "max" => self.max.map_or(0., Scalar::as_f64),
            _ => {
                if let Some(moment) = &self.wide_moment {
                    (**moment).clone().sqrt_divided_by(self.count)
                } else {
                    if !self.moment.is_finite() {
                        return Err(invalid(
                            "Attribute window variance exceeds finite working range",
                        ));
                    }
                    (self.moment.max(0.0) / self.count as f64).sqrt()
                }
            }
        };
        float_output(value)
    }
}
/// Full row-count windows; missing/nonnumeric cells do not count toward the statistic.
/// Output is f64; standard deviation uses the population denominator.
pub fn rolling(input: &[AttrValue], window: usize, operation: &str) -> GraphResult<Vec<AttrValue>> {
    validate(operation)?;
    if window == 0 {
        return Err(invalid("Window size must be greater than zero"));
    }
    // Validate all numeric cells, even when no complete window will be emitted.
    for value in input {
        numeric(value)?;
    }
    let mut output = Vec::with_capacity(input.len());
    for end in 1..=input.len() {
        if end < window {
            output.push(AttrValue::Null);
            continue;
        }
        let mut state = Running::new();
        for value in &input[end - window..end] {
            state.add(value)?;
        }
        output.push(state.output(operation, false)?);
    }
    Ok(output)
}
/// One pass, fixed state; nonnumeric cells carry the previous statistic forward.
pub fn expanding(
    input: &[AttrValue],
    operation: &str,
    empty_sum: bool,
) -> GraphResult<Vec<AttrValue>> {
    validate(operation)?;
    let mut state = Running::new();
    let mut output = Vec::with_capacity(input.len());
    for value in input {
        state.add(value)?;
        output.push(state.output(operation, empty_sum)?);
    }
    Ok(output)
}
/// Fractional change retains integer differences; missing/zero previous cells give null.
pub fn pct_change(input: &[AttrValue], periods: usize) -> GraphResult<Vec<AttrValue>> {
    let mut output = Vec::with_capacity(input.len());
    for (index, current) in input.iter().enumerate() {
        let Some(previous) = index.checked_sub(periods) else {
            output.push(AttrValue::Null);
            continue;
        };
        let current = numeric(current)?;
        let previous = numeric(&input[previous])?;
        let value = match (current, previous) {
            (Some(current), Some(previous)) if previous.as_f64() != 0. => {
                float_output(current.difference(previous) / previous.as_f64())?
            }
            _ => AttrValue::Null,
        };
        output.push(value);
    }
    Ok(output)
}
