//! Deterministic scalar ordering and pagination over captured entity identities.
use super::{EdgeFilter, NodeFilter};
use crate::{
    storage::numeric::scalar::{attribute_number, compare_inputs, NumericInput},
    AttrName, AttrValue, GraphError, GraphResult, GraphView,
};
use std::{borrow::Cow, cmp::Ordering};

/// Ordering and pagination of matching IDs. Missing/Null cells sort last in both
/// directions; equal property values always break ties by ascending entity ID.
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    /// None orders by entity ID; Some orders by a comparable scalar property.
    pub order_by: Option<AttrName>,
    /// Reverse scalar/ID ordering; property ties still use ascending IDs.
    pub descending: bool,
    /// Matching rows skipped after ordering; out-of-range offsets return empty.
    pub offset: usize,
    /// Maximum returned rows; None returns every remaining match.
    pub limit: Option<usize>,
}

impl GraphView {
    /// Scan this immutable selection, then order/page matching node identities.
    /// This explicit scan does not construct indexes or silently recover one.
    pub fn query_nodes(
        &self,
        filter: &NodeFilter,
        options: &QueryOptions,
    ) -> GraphResult<Vec<usize>> {
        let ids =
            super::filtering::nodes(&super::filtering::Frozen(self), &self.node_ids(), filter)?;
        page(self, true, ids, options)
    }
    /// Scan captured edge instances, preserving selection, direction and parallel IDs.
    pub fn query_edges(
        &self,
        filter: &EdgeFilter,
        options: &QueryOptions,
    ) -> GraphResult<Vec<usize>> {
        let ids =
            super::filtering::edges(&super::filtering::Frozen(self), &self.edge_ids(), filter)?;
        page(self, false, ids, options)
    }
}
enum Key<'a> {
    Missing,
    Number(NumericInput),
    Text(Cow<'a, str>),
    Bool(bool),
    Temporal(&'a AttrValue, u8),
}
impl Key<'_> {
    fn domain(&self) -> Option<u8> {
        match self {
            Self::Missing => None,
            Self::Number(_) => Some(0),
            Self::Text(_) => Some(1),
            Self::Bool(_) => Some(2),
            Self::Temporal(_, kind) => Some(*kind),
        }
    }
    fn compare(&self, other: &Self, descending: bool) -> Ordering {
        let ordered = match (self, other) {
            (Self::Missing, Self::Missing) => return Ordering::Equal,
            (Self::Missing, _) => return Ordering::Greater,
            (_, Self::Missing) => return Ordering::Less,
            (Self::Number(a), Self::Number(b)) => compare_inputs(*a, *b),
            (Self::Text(a), Self::Text(b)) => Some(a.cmp(b)),
            (Self::Bool(a), Self::Bool(b)) => Some(a.cmp(b)),
            (Self::Temporal(a, _), Self::Temporal(b, _)) => a.temporal_cmp(b),
            _ => None,
        }
        .unwrap_or(Ordering::Equal); // Complete domain/NaN validation precedes sorting.
        if descending {
            ordered.reverse()
        } else {
            ordered
        }
    }
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn key(value: Option<&AttrValue>) -> GraphResult<Key<'_>> {
    let Some(value) = value else {
        return Ok(Key::Missing);
    };
    if let Some(number) = attribute_number(value) {
        if compare_inputs(number, number).is_none() {
            return Err(invalid("NaN cannot be ordered"));
        }
        return Ok(Key::Number(number));
    }
    Ok(match value {
        AttrValue::Null => Key::Missing,
        AttrValue::Text(v) => Key::Text(Cow::Borrowed(v)),
        AttrValue::CompactText(v) => Key::Text(Cow::Borrowed(v.as_str())),
        AttrValue::CompressedText(v) => {
            Key::Text(Cow::Owned(v.decompress_text().map_err(invalid)?))
        }
        AttrValue::Bool(v) => Key::Bool(*v),
        AttrValue::UtcInstant(_) => Key::Temporal(value, 3),
        AttrValue::Date(_) => Key::Temporal(value, 4),
        AttrValue::Duration(_) => Key::Temporal(value, 5),
        _ => {
            return Err(invalid(
                "Ordering requires a numeric, text, Boolean or temporal scalar",
            ))
        }
    })
}
pub(crate) fn page(
    view: &GraphView,
    node: bool,
    ids: Vec<usize>,
    options: &QueryOptions,
) -> GraphResult<Vec<usize>> {
    if let Some(name) = &options.order_by {
        if name.trim().is_empty() {
            return Err(invalid("order_by must be a nonempty property name"));
        }
        let snapshot = view.read_state().snapshot();
        let attrs = if node {
            &snapshot.node_attributes
        } else {
            &snapshot.edge_attributes
        };
        let mut domain = None;
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            let key = key(attrs.get(&id).and_then(|attrs| attrs.get(name))).map_err(|e| {
                invalid(format!(
                    "Cannot order {} {id} property {name}: {e}",
                    if node { "node" } else { "edge" }
                ))
            })?;
            if let Some(kind) = key.domain() {
                if domain.is_some_and(|old| old != kind) {
                    return Err(invalid(format!(
                        "Cannot order mixed scalar domains for property {name}"
                    )));
                }
                domain = Some(kind);
            }
            rows.push((id, key));
        }
        // Validate every match above, then partition out only the requested page.
        // The ID tie breaker makes the order total, so partitioning and a full
        // sort select identical rows even when many scalar values are equal.
        let start = options.offset.min(rows.len());
        let end = start
            .saturating_add(options.limit.unwrap_or(usize::MAX))
            .min(rows.len());
        if start == end {
            return Ok(Vec::new());
        }
        let mut compare = |(a, av): &(usize, Key<'_>), (b, bv): &(usize, Key<'_>)| {
            av.compare(bv, options.descending).then(a.cmp(b))
        };
        if end < rows.len() {
            rows.select_nth_unstable_by(end, &mut compare);
        }
        if start > 0 {
            rows[..end].select_nth_unstable_by(start, &mut compare);
        }
        let selected = &mut rows[start..end];
        selected.sort_unstable_by(compare);
        return Ok(selected.iter().map(|(id, _)| *id).collect());
    }
    Ok(id_page(&ids, options))
}

/// Page an ascending ID posting without copying/reversing the complete posting.
pub(crate) fn id_page(ids: &[usize], options: &QueryOptions) -> Vec<usize> {
    let start = options.offset.min(ids.len());
    let end = start
        .saturating_add(options.limit.unwrap_or(usize::MAX))
        .min(ids.len());
    if options.descending {
        ids[ids.len() - end..ids.len() - start]
            .iter()
            .rev()
            .copied()
            .collect()
    } else {
        ids[start..end].to_vec()
    }
}
