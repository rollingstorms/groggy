//! Remap typed graph-local entity references during independent materialization.
use crate::{AttrValue, GraphError, GraphResult};
use std::collections::HashMap;

// Plain integer/vector/JSON values are opaque user data. Only explicit reference
// variants have graph identity semantics. Consume the value to preserve buffers.
pub(crate) fn remap(
    mut value: AttrValue,
    nodes: &HashMap<usize, usize>,
    edges: &HashMap<usize, usize>,
    scope: &str,
) -> GraphResult<AttrValue> {
    let (ids, mapping, kind) = match &mut value {
        AttrValue::SubgraphRef(_) => {
            return Err(GraphError::InvalidInput(
                "SubgraphRef requires portable source pool contents".into(),
            ));
        }
        AttrValue::NodeArray(ids) => (ids, nodes, "node"),
        AttrValue::EdgeArray(ids) => (ids, edges, "edge"),
        _ => return Ok(value),
    };
    for id in ids {
        *id = mapping.get(id).copied().ok_or_else(|| {
            GraphError::InvalidInput(format!("referenced {kind} {id} is absent from {scope}"))
        })?;
    }
    Ok(value)
}
