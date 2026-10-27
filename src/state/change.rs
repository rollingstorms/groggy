//! Logical typed property differences shared by snapshots and history.
//!
//! This describes before/after values, independent of column offsets, storage
//! journals, publication clocks and effective-time selection. Absence is `None`;
//! an explicit null is `Some(AttrValue::Null)`. Legacy state/history paths re-export
//! these same types. No wire format or mutation/publication policy is changed.
use crate::{AttrName, AttrValue, EdgeId, GraphResult, NodeId};
mod index;
mod snapshot;
pub use index::IndexChangeView;
pub(crate) use snapshot::{snapshot_changes, snapshot_changes_by};

/// A single property difference between two selected graph states.
#[derive(Debug, Clone)]
pub struct AttributeChange {
    /// Node or edge whose property differs.
    pub entity_type: EntityType,
    /// Existing public diff identity representation, retained without narrowing.
    pub entity_id: u64,
    /// Property name.
    pub attr_name: AttrName,
    /// Before value; absence is distinct from explicit null.
    pub old_value: Option<AttrValue>,
    /// After value; absence is distinct from explicit null.
    pub new_value: Option<AttrValue>,
}

/// Logical property-change target. Diagnostic and sampling scopes are separate.
#[derive(Debug, Clone)]
pub enum EntityType {
    /// A graph node.
    Node,
    /// A graph edge instance.
    Edge,
}

/// A change to the ordered stored endpoints of an existing edge instance.
/// Independent of property changes, clocks and raw storage offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EdgeEndpointChange {
    /// Stable edge instance identity, including parallel edges and self loops.
    pub edge_id: crate::EdgeId,
    /// Endpoints in the source selected state.
    pub old_endpoints: (crate::NodeId, crate::NodeId),
    /// Endpoints in the destination selected state.
    pub new_endpoints: (crate::NodeId, crate::NodeId),
}

/// A set of changes that can be committed to history
#[derive(Debug, Clone)]
pub struct ChangeSet {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    pub edges_removed: Vec<EdgeId>,
    /// Stable edge identities whose endpoints changed.
    pub edge_endpoint_changes: Vec<crate::EdgeEndpointChange>,
    pub node_attr_changes: Vec<(NodeId, AttrName, Option<AttrValue>, Option<AttrValue>)>,
    pub edge_attr_changes: Vec<(EdgeId, AttrName, Option<AttrValue>, Option<AttrValue>)>,
}

impl ChangeSet {
    /// Create an empty change set
    pub fn new() -> Self {
        Self {
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            edge_endpoint_changes: Vec::new(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        }
    }

    /// Check if the change set is empty
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
            && self.edge_endpoint_changes.is_empty()
            && self.node_attr_changes.is_empty()
            && self.edge_attr_changes.is_empty()
    }
}

/// Compare exact retained property payloads, normalizing only supported storage
/// representations. Distinct floating widths, bits and NaN payloads stay distinct.
/// This is intentionally separate from AttrValue's logical ordering/equality.
pub(crate) fn same_value(a: &AttrValue, b: &AttrValue) -> GraphResult<bool> {
    use AttrValue::*;
    Ok(match (a, b) {
        (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
        (Float64(a), Float64(b)) => a.to_bits() == b.to_bits(),
        (FloatVec(a), FloatVec(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (Float64Vec(a), Float64Vec(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (Int(a), Int(b)) => a == b,
        (SmallInt(a), SmallInt(b)) => a == b,
        (Int(a), SmallInt(b)) | (SmallInt(b), Int(a)) => *a == i64::from(*b),
        (Text(a), Text(b)) => a == b,
        (IntVec(a), IntVec(b)) => a == b,
        (TextVec(a), TextVec(b)) => a == b,
        (BoolVec(a), BoolVec(b)) => a == b,
        (Bool(a), Bool(b)) => a == b,
        (Bytes(a), Bytes(b)) => a == b,
        (NodeArray(a), NodeArray(b)) | (EdgeArray(a), EdgeArray(b)) => a == b,
        (SubgraphRef(a), SubgraphRef(b)) => a == b,
        (UtcInstant(a), UtcInstant(b)) => a == b,
        (Date(a), Date(b)) => a == b,
        (Duration(a), Duration(b)) => a == b,
        (Null, Null) => true,
        _ => {
            crate::storage::retained_group::encode(a)? == crate::storage::retained_group::encode(b)?
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CalendarDate, CompactString, CompressedData, FixedDuration, UtcInstant};
    #[test]
    fn exact_value_fast_paths_match_portable_codec_for_every_supported_value_family() {
        use AttrValue as V;
        let values = vec![
            V::Float(0.0),
            V::Float(-0.0),
            V::Float(f32::from_bits(0x7fc00001)),
            V::Float(f32::from_bits(0x7fc00002)),
            V::Float64(0.0),
            V::Float64(-0.0),
            V::Float64(f64::from_bits(0x7ff8000000000001)),
            V::Float64(f64::from_bits(0x7ff8000000000002)),
            V::FloatVec(vec![0.0, -0.0, f32::from_bits(0x7fc00001)]),
            V::FloatVec(vec![0.0, -0.0, f32::from_bits(0x7fc00002)]),
            V::Float64Vec(vec![0.0, -0.0, f64::from_bits(0x7ff8000000000001)]),
            V::Float64Vec(vec![0.0, -0.0, f64::from_bits(0x7ff8000000000002)]),
            V::Int(i64::MAX),
            V::Int(1),
            V::SmallInt(1),
            V::IntVec(vec![1, 2]),
            V::Text("text".into()),
            V::CompactText(CompactString::new("text")),
            V::CompressedText(CompressedData::compress_text("text")),
            V::CompressedFloatVec(CompressedData::compress_float_vec(&[1.0, -0.0])),
            V::FloatVec(vec![1.0, -0.0]),
            V::TextVec(vec!["text".into()]),
            V::Bool(true),
            V::BoolVec(vec![true, false]),
            V::Bytes(vec![0, 255]),
            V::Null,
            V::SubgraphRef(17),
            V::NodeArray(vec![1, 2]),
            V::EdgeArray(vec![1, 2]),
            V::Json("{\"a\":1}".into()),
            V::UtcInstant(UtcInstant::new(0, 1).unwrap()),
            V::Date(CalendarDate::new(2026, 10, 27).unwrap()),
            V::Duration(FixedDuration::new(1, 0, 0).unwrap()),
        ];
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                let expected = crate::storage::retained_group::encode(a).unwrap()
                    == crate::storage::retained_group::encode(b).unwrap();
                assert_eq!(same_value(a, b).unwrap(), expected, "pair {i},{j}");
            }
        }
        assert!(same_value(&V::Json("{".into()), &V::Json("{".into())).is_err());
    }
}
