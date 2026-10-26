//! Immutable group revisions and explicit portable source payloads.
use crate::{
    state::{GraphSnapshot, PinnedReadState},
    types::{GraphType, SubgraphId},
    AttrValue, GraphError, GraphResult, GraphView,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};
#[derive(Debug, Clone)]
pub struct RetainedGroup {
    pub(crate) kind: String,
    pub(crate) view: GraphView,
    id: SubgraphId,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupDocument {
    version: u32,
    kind: String,
    state_id: u64,
    graph_type: GraphType,
    read_time: Option<crate::state::effective_read::ReadTime>,
    #[serde(default)]
    aliases: BTreeMap<SubgraphId, SubgraphId>,
    nodes: Vec<usize>,
    edges: BTreeMap<usize, (usize, usize)>,
    node_attributes: BTreeMap<usize, BTreeMap<String, Value>>,
    edge_attributes: BTreeMap<usize, BTreeMap<String, Value>>,
}
/// Reference tags are accepted only inside formats that carry their source groups.
pub(crate) fn encode(value: &AttrValue) -> GraphResult<Value> {
    if let AttrValue::SubgraphRef(id) = value {
        return Ok(serde_json::json!({"type":"subgraph_ref","value":id}));
    }
    super::value_codec::encode(value)
}
pub(crate) fn decode(value: Value) -> GraphResult<AttrValue> {
    if value.get("type").and_then(Value::as_str) == Some("subgraph_ref") {
        if value.as_object().is_none_or(|o| o.len() != 2) {
            return Err(invalid("Malformed group reference"));
        }
        let id = value
            .get("value")
            .and_then(Value::as_u64)
            .and_then(|id| usize::try_from(id).ok())
            .ok_or_else(|| invalid("Invalid group reference ID"))?;
        return Ok(AttrValue::SubgraphRef(id));
    }
    super::value_codec::decode(value)
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
pub(crate) fn references(snapshot: &GraphSnapshot) -> HashSet<SubgraphId> {
    snapshot
        .node_attributes
        .values()
        .chain(snapshot.edge_attributes.values())
        .flat_map(|row| row.values())
        .filter_map(|value| {
            if let AttrValue::SubgraphRef(id) = value {
                Some(*id)
            } else {
                None
            }
        })
        .collect()
}
type EncodedAttributes = BTreeMap<usize, BTreeMap<String, Value>>;
fn encode_attrs(
    values: HashMap<usize, HashMap<String, AttrValue>>,
) -> GraphResult<EncodedAttributes> {
    values
        .into_iter()
        .map(|(id, row)| {
            let row = row
                .into_iter()
                .map(|(name, value)| Ok((name, encode(&value)?)))
                .collect::<GraphResult<_>>()?;
            Ok((id, row))
        })
        .collect()
}
fn decode_attrs(
    values: EncodedAttributes,
) -> GraphResult<HashMap<usize, HashMap<String, AttrValue>>> {
    values
        .into_iter()
        .map(|(id, row)| {
            let row = row
                .into_iter()
                .map(|(name, value)| Ok((name, decode(value)?)))
                .collect::<GraphResult<_>>()?;
            Ok((id, row))
        })
        .collect()
}
impl RetainedGroup {
    /// Normalize a frozen selection without retaining its full parent graph.
    pub fn new(view: GraphView, kind: String) -> GraphResult<Self> {
        let snapshot = view.to_snapshot();
        let mut groups = HashMap::new();
        for id in references(&snapshot) {
            let group =
                view.read_state().groups().get(&id).ok_or_else(|| {
                    invalid("Retained group needs frozen nested source revisions")
                })?;
            groups.insert(id, group.clone());
        }
        // Store only selected rows and directly referenced groups, without retaining
        // the full parent graph or granting writeback authority to a group revision.
        let mut state = PinnedReadState::from_snapshot(snapshot)?.with_groups(Arc::new(groups));
        if let Some(time) = view.read_state().read_time() {
            state = state.with_read_time(time.clone());
        }
        let mut group = Self {
            kind,
            view: GraphView::all(state),
            id: 0,
        };
        let encoded = serde_json::to_vec(&group.document()?)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        group.id = usize::try_from(xxhash_rust::xxh3::xxh3_64(&encoded))
            .map_err(|_| invalid("Group reference exceeds platform ID width"))?;
        Ok(group)
    }
    /// Content-addressed revision ID.
    pub fn id(&self) -> SubgraphId {
        self.id
    }
    pub(crate) fn same_content(&self, other: &Self) -> GraphResult<bool> {
        Ok(self.document()? == other.document()?)
    }
    pub(crate) fn document(&self) -> GraphResult<GroupDocument> {
        let snapshot = self.view.to_snapshot();
        Ok(GroupDocument {
            version: 1,
            kind: self.kind.clone(),
            state_id: snapshot.state_id,
            graph_type: snapshot.graph_type,
            read_time: self.view.read_state().read_time().cloned(),
            aliases: self
                .view
                .read_state()
                .groups()
                .iter()
                .filter_map(|(&id, group)| (id != group.id()).then_some((id, group.id())))
                .collect(),
            nodes: snapshot.active_nodes,
            edges: snapshot.edges.into_iter().collect(),
            node_attributes: encode_attrs(snapshot.node_attributes)?,
            edge_attributes: encode_attrs(snapshot.edge_attributes)?,
        })
    }
}
impl GroupDocument {
    pub(crate) fn restore(
        self,
        available: &HashMap<SubgraphId, RetainedGroup>,
    ) -> GraphResult<RetainedGroup> {
        if self.version != 1 {
            return Err(invalid("Unsupported retained group format version"));
        }
        let snapshot = GraphSnapshot {
            graph_type: self.graph_type,
            state_id: self.state_id,
            active_nodes: self.nodes,
            edges: self.edges.into_iter().collect(),
            node_attributes: decode_attrs(self.node_attributes)?,
            edge_attributes: decode_attrs(self.edge_attributes)?,
        };
        let node_ids: HashSet<_> = snapshot.active_nodes.iter().copied().collect();
        if snapshot
            .node_attributes
            .keys()
            .any(|id| !node_ids.contains(id))
            || snapshot
                .edge_attributes
                .keys()
                .any(|id| !snapshot.edges.contains_key(id))
        {
            return Err(invalid("Group attributes contain out-of-scope IDs"));
        }
        let refs = references(&snapshot);
        if self.aliases.keys().any(|id| !refs.contains(id)) {
            return Err(invalid("Unused legacy group source alias"));
        }
        let groups = refs
            .into_iter()
            .map(|id| {
                let canonical = self.aliases.get(&id).copied().unwrap_or(id);
                available
                    .get(&canonical)
                    .cloned()
                    .map(|group| (id, group))
                    .ok_or_else(|| invalid("Missing or cyclic retained group dependency"))
            })
            .collect::<GraphResult<_>>()?;
        let mut state = PinnedReadState::from_snapshot(snapshot)?.with_groups(Arc::new(groups));
        if let Some(time) = self.read_time {
            state = state.with_read_time(time);
        }
        RetainedGroup::new(GraphView::all(state), self.kind)
    }
}
/// Dependency order, without recursion or duplicate group payloads.
pub(crate) fn documents(root: &RetainedGroup) -> GraphResult<Vec<(SubgraphId, GroupDocument)>> {
    let mut stack = vec![(root.clone(), false)];
    let mut done = HashSet::new();
    let mut visiting = HashSet::new();
    let mut output = Vec::new();
    while let Some((group, expanded)) = stack.pop() {
        let id = group.id();
        if done.contains(&id) {
            continue;
        }
        if expanded {
            visiting.remove(&id);
            done.insert(id);
            output.push((id, group.document()?));
            continue;
        }
        if !visiting.insert(id) {
            return Err(invalid("Cyclic retained group dependency"));
        }
        stack.push((group.clone(), true));
        let mut dependencies: Vec<_> = group.view.read_state().groups().values().cloned().collect();
        dependencies.sort_by_key(RetainedGroup::id);
        for child in dependencies.into_iter().rev() {
            if !done.contains(&child.id()) {
                stack.push((child, false));
            }
        }
    }
    Ok(output)
}
