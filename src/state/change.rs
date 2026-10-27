//! Logical typed property differences shared by snapshots and history.
//!
//! This describes before/after values, independent of column offsets, storage
//! journals, publication clocks and effective-time selection. Absence is `None`;
//! an explicit null is `Some(AttrValue::Null)`. Legacy state/history paths re-export
//! these same types. No wire format or mutation/publication policy is changed.
use crate::{AttrName, AttrValue};

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
