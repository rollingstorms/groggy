//! Read-only entity/column adapters. GraphView remains the sole state and membership owner.
use super::GraphView;
use crate::{
    storage::array::BaseArray, AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId,
};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, OnceLock},
};
#[derive(Debug, Clone, Copy)]
enum Kind {
    Node,
    Edge,
}
/// Read-only entity bound to its exact captured graph selection.
#[derive(Debug, Clone)]
pub struct GraphViewEntity {
    view: GraphView,
    id: usize,
    kind: Kind,
}
impl GraphViewEntity {
    /// Original entity ID, independent of collection position.
    pub fn id(&self) -> usize {
        self.id
    }
    /// Whether this handle represents a node rather than an edge.
    pub fn is_node(&self) -> bool {
        matches!(self.kind, Kind::Node)
    }
    fn attributes(&self) -> Option<&HashMap<AttrName, AttrValue>> {
        attributes(&self.view, self.kind, self.id)
    }
    /// Captured property, distinguishing absence from explicit null.
    pub fn get(&self, name: &AttrName) -> Option<AttrValue> {
        self.attributes().and_then(|attrs| attrs.get(name)).cloned()
    }
    /// Captured property presence; explicit null is present.
    pub fn contains_attribute(&self, name: &AttrName) -> bool {
        self.attributes()
            .is_some_and(|attrs| attrs.contains_key(name))
    }
    /// Sorted present keys only.
    pub fn keys(&self) -> Vec<AttrName> {
        self.attributes()
            .into_iter()
            .flat_map(|attrs| attrs.keys())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    /// Detached typed properties; mutating this map cannot change the capture.
    pub fn to_dict(&self) -> BTreeMap<AttrName, AttrValue> {
        self.attributes()
            .into_iter()
            .flat_map(|attrs| attrs.iter())
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }
    /// Selected frozen node degree, with canonical loop/parallel conventions.
    pub fn degree(&self) -> GraphResult<usize> {
        match self.kind {
            Kind::Node => self.view.degree(self.id),
            Kind::Edge => Err(invalid("degree is defined only for nodes")),
        }
    }
    /// Selected outgoing neighbor occurrences, not owner-wide neighbors.
    pub fn neighbors(&self) -> GraphResult<Vec<NodeId>> {
        match self.kind {
            Kind::Node => self.view.neighbors(self.id),
            Kind::Edge => Err(invalid("neighbors is defined only for nodes")),
        }
    }
    /// Captured edge endpoints.
    pub fn endpoints(&self) -> GraphResult<(NodeId, NodeId)> {
        match self.kind {
            Kind::Edge => self.view.edge_endpoints(self.id),
            Kind::Node => Err(invalid("endpoints is defined only for edges")),
        }
    }
    /// Frozen source group for a captured meta-node.
    pub fn meta_subgraph(&self) -> GraphResult<GraphView> {
        match self.kind {
            Kind::Node => self.view.meta_subgraph(self.id),
            Kind::Edge => Err(invalid("meta_subgraph is defined only for nodes")),
        }
    }
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn attributes(view: &GraphView, kind: Kind, id: usize) -> Option<&HashMap<AttrName, AttrValue>> {
    match kind {
        Kind::Node => view.read_state().snapshot().node_attributes.get(&id),
        Kind::Edge => view.read_state().snapshot().edge_attributes.get(&id),
    }
}
/// Homogeneous captured entity access, with stable ID order and owned column results.
#[derive(Debug, Clone)]
pub struct GraphViewEntities {
    view: GraphView,
    kind: Kind,
    ids: Arc<OnceLock<Arc<[usize]>>>,
}
impl GraphViewEntities {
    /// Ascending original IDs; never floats or positional aliases.
    pub fn ids(&self) -> &[usize] {
        self.ids.get_or_init(|| match self.kind {
            Kind::Node => self.view.node_ids().into(),
            Kind::Edge => self.view.edge_ids().into(),
        })
    }
    /// Number of selected entities.
    pub fn len(&self) -> usize {
        match self.kind {
            Kind::Node => self.view.node_count(),
            Kind::Edge => self.view.edge_count(),
        }
    }
    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Whether this collection contains nodes.
    pub fn is_nodes(&self) -> bool {
        matches!(self.kind, Kind::Node)
    }
    /// Membership by original ID, without materializing positional order.
    pub fn contains_id(&self, id: usize) -> bool {
        match self.kind {
            Kind::Node => self.view.contains_node(id),
            Kind::Edge => self.view.contains_edge(id),
        }
    }
    /// Retrieve an entity by ID; the ID must belong to this exact selection.
    pub fn by_id(&self, id: usize) -> GraphResult<GraphViewEntity> {
        match self.kind {
            Kind::Node => self.view.node(id),
            Kind::Edge => self.view.edge(id),
        }
    }
    /// Retrieve a position in ascending ID order; negative positions count from the end.
    pub fn at(&self, position: isize) -> GraphResult<GraphViewEntity> {
        let position = if position < 0 {
            self.len().checked_sub(position.unsigned_abs())
        } else {
            Some(position as usize)
        };
        let id = position
            .and_then(|position| self.ids().get(position))
            .copied()
            .ok_or_else(|| invalid("Entity position is out of range"))?;
        Ok(GraphViewEntity {
            view: self.view.clone(),
            id,
            kind: self.kind,
        })
    }
    /// All nodes retain the exact parent selection; all edges retain their endpoints only.
    pub fn all(&self) -> GraphResult<GraphView> {
        match self.kind {
            Kind::Node => Ok(self.view.clone()),
            Kind::Edge => self.view.select_edges(self.ids()),
        }
    }
    /// Select explicit original IDs within this collection; duplicates coalesce.
    pub fn select(&self, ids: &[usize]) -> GraphResult<GraphView> {
        match self.kind {
            Kind::Node => self.view.select_nodes(ids),
            Kind::Edge => self.view.select_edges(ids),
        }
    }
    /// Select a translated positional slice. Bounds are checked before returning a child.
    pub fn slice_positions(
        &self,
        start: isize,
        step: isize,
        count: usize,
    ) -> GraphResult<GraphView> {
        if step == 0 {
            return Err(invalid("Slice step must be nonzero"));
        }
        if count > self.len() {
            return Err(invalid("Slice length exceeds entity count"));
        }
        let mut ids = Vec::with_capacity(count);
        let mut position = start;
        for index in 0..count {
            let selected = usize::try_from(position)
                .ok()
                .and_then(|position| self.ids().get(position))
                .copied()
                .ok_or_else(|| invalid("Slice position is out of range"))?;
            ids.push(selected);
            if index + 1 < count {
                position = position
                    .checked_add(step)
                    .ok_or_else(|| invalid("Slice position overflow"))?;
            }
        }
        self.select(&ids)
    }
    /// Aligned owned property column; missing cells are null, with presence available separately.
    pub fn column(&self, name: &AttrName) -> BaseArray<AttrValue> {
        self.project_column(|id| {
            attributes(&self.view, self.kind, id)
                .and_then(|attrs| attrs.get(name))
                .cloned()
                .unwrap_or(AttrValue::Null)
        })
    }
    /// Aligned explicit presence bits; null remains present.
    pub fn presence(&self, name: &AttrName) -> BaseArray<AttrValue> {
        self.project_column(|id| {
            AttrValue::Bool(
                attributes(&self.view, self.kind, id).is_some_and(|attrs| attrs.contains_key(name)),
            )
        })
    }
    fn project_column(&self, project: impl Fn(usize) -> AttrValue + Sync) -> BaseArray<AttrValue> {
        let ids = self.ids();
        BaseArray::new(if ids.len() > 1000 {
            ids.par_iter().map(|&id| project(id)).collect()
        } else {
            ids.iter().map(|&id| project(id)).collect()
        })
    }
    /// Whether any selected entity has this property, including explicit null.
    pub fn has_column(&self, name: &AttrName) -> bool {
        let ids = match self.kind {
            Kind::Node => self.view.nodes(),
            Kind::Edge => self.view.edges(),
        };
        ids.iter().any(|&id| {
            attributes(&self.view, self.kind, id).is_some_and(|attrs| attrs.contains_key(name))
        })
    }
    /// All present column names in canonical sorted order.
    pub fn attribute_names(&self) -> Vec<AttrName> {
        self.ids()
            .iter()
            .filter_map(|&id| attributes(&self.view, self.kind, id))
            .flat_map(|attrs| attrs.keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .cloned()
            .collect()
    }
    /// Select an exact-length, boolean-only aligned mask; empty results are valid views.
    pub fn select_mask(&self, mask: &BaseArray<AttrValue>) -> GraphResult<GraphView> {
        if mask.len() != self.len() {
            return Err(invalid(
                "Boolean mask length must match the captured entity collection",
            ));
        }
        let mut ids = Vec::new();
        for (&id, value) in self.ids().iter().zip(mask.iter()) {
            match value {
                AttrValue::Bool(true) => ids.push(id),
                AttrValue::Bool(false) => {}
                _ => return Err(invalid("Entity masks must contain only booleans")),
            }
        }
        self.select(&ids)
    }
    /// Filter selected captured nodes.
    pub fn filter_nodes(&self, filter: &crate::query::NodeFilter) -> GraphResult<GraphView> {
        if !self.is_nodes() {
            return Err(invalid("A node filter requires a node collection"));
        }
        self.view.filter_nodes(filter)
    }
    /// Filter selected captured edge instances.
    pub fn filter_edges(&self, filter: &crate::query::EdgeFilter) -> GraphResult<GraphView> {
        if self.is_nodes() {
            return Err(invalid("An edge filter requires an edge collection"));
        }
        self.view.filter_edges(filter)
    }
    /// Cheap iterator retaining the capture and stable ID order.
    pub fn iter(&self) -> GraphViewEntityIterator {
        let _ = self.ids();
        GraphViewEntityIterator {
            entities: self.clone(),
            position: 0,
        }
    }
}
/// Captured entity iterator; it borrows no working graph or Python owner.
#[derive(Debug, Clone)]
pub struct GraphViewEntityIterator {
    entities: GraphViewEntities,
    position: usize,
}
impl Iterator for GraphViewEntityIterator {
    type Item = GraphViewEntity;
    fn next(&mut self) -> Option<Self::Item> {
        let id = *self.entities.ids().get(self.position)?;
        self.position += 1;
        Some(GraphViewEntity {
            view: self.entities.view.clone(),
            id,
            kind: self.entities.kind,
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.entities.len() - self.position;
        (left, Some(left))
    }
}
impl ExactSizeIterator for GraphViewEntityIterator {}
impl GraphView {
    /// Capture a read-only node handle in this exact selection.
    pub fn node(&self, id: NodeId) -> GraphResult<GraphViewEntity> {
        if !self.contains_node(id) {
            return Err(invalid("Node is outside the captured selection"));
        }
        Ok(GraphViewEntity {
            view: self.clone(),
            id,
            kind: Kind::Node,
        })
    }
    /// Capture a read-only edge handle in this exact selection.
    pub fn edge(&self, id: EdgeId) -> GraphResult<GraphViewEntity> {
        self.edge_endpoints(id)?;
        Ok(GraphViewEntity {
            view: self.clone(),
            id,
            kind: Kind::Edge,
        })
    }
    /// Read-only nodes and aligned columns in ascending original ID order.
    pub fn node_accessor(&self) -> GraphViewEntities {
        GraphViewEntities {
            view: self.clone(),
            kind: Kind::Node,
            ids: Arc::new(OnceLock::new()),
        }
    }
    /// Read-only edge instances and aligned columns in ascending original ID order.
    pub fn edge_accessor(&self) -> GraphViewEntities {
        GraphViewEntities {
            view: self.clone(),
            kind: Kind::Edge,
            ids: Arc::new(OnceLock::new()),
        }
    }
}
