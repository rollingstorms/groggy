//! Table-specific missing/output adapters over checked numeric column kernels.
use crate::{
    storage::{
        array::BaseArray,
        numeric::{
            attributes::{self, Value},
            dot_product::Dot,
            reductions::Scalar,
        },
    },
    AttrValue, GraphError, GraphResult,
};
use std::cmp::Ordering;
pub(super) fn compare(a: &AttrValue, b: &AttrValue) -> Ordering {
    match (Value::parse(a), Value::parse(b)) {
        (Some(a), Some(b)) if !a.as_f64().is_nan() && !b.as_f64().is_nan() => a.numeric_cmp(b),
        _ => a.cmp(b),
    }
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
pub(super) fn aggregate(
    input: &[&AttrValue],
    operation: &str,
    pivot: bool,
) -> GraphResult<AttrValue> {
    if !["count", "sum", "mean", "avg", "min", "max"].contains(&operation) {
        return Err(invalid("Unsupported table aggregation"));
    }
    if operation == "count" {
        return i64::try_from(input.len())
            .map(AttrValue::Int)
            .map_err(|_| invalid("Table count exceeds i64"));
    }
    if input.is_empty() {
        return Ok(AttrValue::Null);
    }
    if pivot && ["min", "max"].contains(&operation) {
        // Preserve original scalar output; use actual value identity/order for text too.
        let mut selected = input[0];
        for &value in input {
            if let Some(number) = Value::parse(value) {
                if !number.as_f64().is_finite() {
                    return Err(invalid("Table extrema require finite numbers"));
                }
            }
            let order = compare(value, selected);
            if (operation == "min" && order == Ordering::Less)
                || (operation == "max" && order == Ordering::Greater)
            {
                selected = value;
            }
        }
        return Ok(selected.clone());
    }
    let mut numeric = Vec::with_capacity(input.len());
    for &value in input {
        if Value::parse(value).is_some() {
            numeric.push(value.clone())
        } else if pivot && !matches!(value, AttrValue::Null) {
            return Err(invalid("Pivot sum/mean require numeric or null cells"));
        }
    }
    if numeric.is_empty() {
        return Ok(if pivot && operation == "sum" {
            AttrValue::Float(0.)
        } else {
            AttrValue::Null
        });
    }
    let column = BaseArray::new(numeric);
    match operation {
        "sum" if pivot => {
            let values = attributes::values(column.data())?;
            let mut sum = Dot::new();
            for value in values {
                let (hi, lo) = value.parts();
                sum.add(hi, 1.);
                sum.add(lo, 1.);
            }
            attributes::float_output(sum.total())
        }
        "sum" => column.sum(),
        "mean" | "avg" => attributes::float_output(column.mean()?),
        "min" => column.min(),
        _ => column.max(),
    }
}
