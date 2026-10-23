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
            AttrValue::Float64(0.)
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

/// Numeric subset of a profile row. Extrema and rank quartiles retain source types.
pub(super) struct NumericProfile {
    pub mean: AttrValue,
    pub median: AttrValue,
    pub std: AttrValue,
    pub min: AttrValue,
    pub max: AttrValue,
    pub q25: AttrValue,
    pub q75: AttrValue,
}

/// Preserve profile's population dispersion and floor(n*q) rank convention.
/// Text, vectors, Boolean and null cells are not numeric observations.
pub(super) fn profile(input: &[AttrValue]) -> GraphResult<Option<NumericProfile>> {
    let mut values = Vec::new();
    for source in input {
        if let Some(number) = Value::parse(source) {
            if !number.as_f64().is_finite() {
                return Err(invalid(
                    "Table profile requires finite numeric observations",
                ));
            }
            values.push((source, number));
        }
    }
    if values.is_empty() {
        return Ok(None);
    }
    let mean = crate::storage::array::reductions::mean_iter(values.iter().map(|(v, _)| *v))?;
    let std = crate::storage::numeric::reductions::population_std_iter(
        values.iter().map(|(_, number)| *number),
    )
    .ok_or_else(|| invalid("Table profile has no numeric observations"))?;
    values.sort_by(|(_, a), (_, b)| a.numeric_cmp(*b));
    let n = values.len();
    let middle = n / 2;
    let median = if n.is_multiple_of(2) {
        let mut sum = Dot::new();
        for index in [middle - 1, middle] {
            let (hi, lo) = values[index].1.parts();
            sum.add(hi, 0.5);
            sum.add(lo, 0.5);
        }
        attributes::float_output(sum.total())?
    } else {
        attributes::float_output(values[middle].1.as_f64())?
    };
    Ok(Some(NumericProfile {
        mean: attributes::float_output(mean)?,
        median,
        std: attributes::float_output(std)?,
        min: values[0].0.clone(),
        max: values[n - 1].0.clone(),
        q25: values[n / 4].0.clone(),
        q75: values[n - 1 - (n - 1) / 4].0.clone(),
    }))
}
