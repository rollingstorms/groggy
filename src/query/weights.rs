//! Shared scalar-weight rules for checked Dijkstra entry points.
use crate::{AttrValue, EdgeId, GraphError, GraphResult};

pub(crate) fn dijkstra_weight(value: Option<&AttrValue>, edge: EdgeId) -> GraphResult<f64> {
    let weight = match value {
        None => 1.0,
        Some(AttrValue::Float(value)) => *value as f64,
        Some(AttrValue::Int(value)) => *value as f64,
        Some(AttrValue::SmallInt(value)) => *value as f64,
        Some(_) => {
            return Err(GraphError::InvalidInput(format!(
                "edge {edge}: Dijkstra weight must be numeric"
            )))
        }
    };
    if !weight.is_finite() || weight < 0.0 {
        return Err(GraphError::InvalidInput(format!(
            "edge {edge}: Dijkstra weight must be finite and nonnegative, got {weight}"
        )));
    }
    Ok(weight)
}
