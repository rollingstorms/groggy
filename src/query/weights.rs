//! Shared scalar-weight rules for checked Dijkstra entry points.
use crate::{AttrValue, EdgeId, GraphError, GraphResult};

pub(crate) fn dijkstra_weight(value: Option<&AttrValue>, edge: EdgeId) -> GraphResult<f64> {
    let weight = match value {
        None => 1.0,
        Some(AttrValue::Float(value)) => *value as f64,
        Some(AttrValue::Float64(value)) => *value,
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

/// Validate every selected weight and retain the minimum per traversable pair.
/// Callers establish active/captured membership and selected endpoint closure.
pub(crate) fn checked_pairs<'a>(
    edges: &[EdgeId],
    directed: bool,
    mut endpoints: impl FnMut(EdgeId) -> GraphResult<(crate::NodeId, crate::NodeId)>,
    mut value: impl FnMut(EdgeId) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<std::collections::HashMap<(crate::NodeId, crate::NodeId), f64>> {
    let mut weights: std::collections::HashMap<_, f64> = std::collections::HashMap::new();
    for &edge in edges {
        let (u, v) = endpoints(edge)?;
        let weight = dijkstra_weight(value(edge)?, edge)?;
        weights
            .entry((u, v))
            .and_modify(|old| *old = old.min(weight))
            .or_insert(weight);
        if !directed {
            weights
                .entry((v, u))
                .and_modify(|old| *old = old.min(weight))
                .or_insert(weight);
        }
    }
    Ok(weights)
}

/// Shared A* heuristic domain: absent is zero, present is finite nonnegative numeric.
pub(crate) fn checked_heuristics<'a>(
    nodes: &[crate::NodeId],
    mut value: impl FnMut(crate::NodeId) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<std::collections::HashMap<crate::NodeId, f64>> {
    nodes
        .iter()
        .map(|&node| {
            let cost = match value(node)? {
                None => 0.0,
                Some(AttrValue::Float(v)) => f64::from(*v),
                Some(AttrValue::Float64(v)) => *v,
                Some(AttrValue::Int(v)) => *v as f64,
                Some(AttrValue::SmallInt(v)) => f64::from(*v),
                _ => {
                    return Err(GraphError::InvalidInput(format!(
                        "heuristic at node {node} must be numeric"
                    )))
                }
            };
            if !cost.is_finite() || cost < 0.0 {
                return Err(GraphError::InvalidInput(format!(
                    "heuristic at node {node} must be finite and nonnegative"
                )));
            }
            Ok((node, cost))
        })
        .collect()
}
