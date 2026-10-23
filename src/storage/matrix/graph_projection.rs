//! Shared live/frozen graph projections; readers supply value ownership and scope.
use super::{FromAttrValue, GraphMatrix, NumericType};
use crate::{AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId};
use std::collections::HashSet;
fn numeric(value: &AttrValue) -> bool {
    matches!(
        value,
        AttrValue::Int(_) | AttrValue::SmallInt(_) | AttrValue::Float(_) | AttrValue::Bool(_)
    )
}
pub(crate) fn features<'a, T: NumericType + FromAttrValue<T>>(
    nodes: &[NodeId],
    mut names: Vec<AttrName>,
    mut value: impl FnMut(NodeId, &AttrName) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<GraphMatrix<T>> {
    names.sort_unstable();
    names.dedup();
    let mut columns = Vec::new();
    for name in names {
        let mut is_numeric = false;
        // Do not short circuit: every mapped referent must be valid.
        for &node in nodes {
            if let Some(value) = value(node, &name)? {
                is_numeric |= numeric(value);
            }
        }
        if is_numeric {
            columns.push(name);
        }
    }
    let count = nodes
        .len()
        .checked_mul(columns.len())
        .ok_or_else(|| GraphError::InvalidInput("Feature matrix shape overflow".into()))?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
    for &node in nodes {
        for name in &columns {
            data.push(match value(node, name)? {
                Some(value) if numeric(value) => T::from_attr_value(value)?,
                _ => T::from_attr_value(&AttrValue::Null)?,
            });
        }
    }
    let mut matrix =
        GraphMatrix::from_row_major_data(data, nodes.len(), columns.len(), Some(nodes))?;
    matrix.set_column_names(columns);
    Ok(matrix)
}
pub(crate) fn adjacency<'a, T: NumericType + FromAttrValue<T>>(
    nodes: &[NodeId],
    edges: &[EdgeId],
    sparse: bool,
    weight_attr: Option<&AttrName>,
    directed: bool,
    mut endpoints: impl FnMut(EdgeId) -> GraphResult<(NodeId, NodeId)>,
    mut value: impl FnMut(EdgeId, &AttrName) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<GraphMatrix<T>> {
    let mut seen = HashSet::new();
    GraphMatrix::adjacency_from_edge_iter(
        nodes,
        edges.iter().map(|&edge| {
            if !seen.insert(edge) {
                return Err(GraphError::InvalidInput(format!(
                    "Duplicate adjacency edge ID {edge}"
                )));
            }
            let (source, target) = endpoints(edge)?;
            let weight = match weight_attr {
                Some(name) => match value(edge, name)? {
                    Some(value) => T::from_attr_value(value)?,
                    None => T::one(),
                },
                None => T::one(),
            };
            Ok((source, target, weight))
        }),
        directed,
        sparse,
    )
}
