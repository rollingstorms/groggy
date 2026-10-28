//! Explicit retained-revision membership and reverse-reference postings.
use crate::{types::SubgraphId, AttrValue, GraphError, GraphResult, GraphView};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

/// A selected root entity's property referring to a frozen group revision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupReference {
    pub entity_id: usize,
    pub property: String,
}

#[derive(Debug, Default)]
struct GroupData {
    groups: HashMap<SubgraphId, GraphView>,
    roots: BTreeSet<SubgraphId>,
    children: HashMap<SubgraphId, BTreeSet<SubgraphId>>,
    parents: HashMap<SubgraphId, BTreeSet<SubgraphId>>,
    nodes: HashMap<SubgraphId, Vec<GroupReference>>,
    edges: HashMap<SubgraphId, Vec<GroupReference>>,
}

/// Immutable index of selected references and their reachable frozen revisions.
/// Membership is always qualified by revision; unrelated source-local entity IDs
/// are never joined. Builds/rebuilds are explicit and publish no graph changes.
#[derive(Debug, Clone)]
pub struct GroupMembershipIndex {
    view: GraphView,
    data: Arc<GroupData>,
}

type ResolvedReference = (bool, GroupReference, SubgraphId, GraphView);

fn references(view: &GraphView) -> GraphResult<Vec<ResolvedReference>> {
    let mut result = Vec::new();
    for (node, ids) in [(true, view.node_ids()), (false, view.edge_ids())] {
        let snapshot = view.read_state().snapshot();
        let attributes = if node {
            &snapshot.node_attributes
        } else {
            &snapshot.edge_attributes
        };
        for id in ids {
            for (name, value) in attributes.get(&id).into_iter().flatten() {
                if let AttrValue::SubgraphRef(local) = value {
                    let group = view.read_state().groups().get(local).ok_or_else(|| {
                        GraphError::InvalidInput(format!("Group {local} has no frozen source revision; pin an explicit source view"))
                    })?;
                    // Legacy aliases are local to this source context. Resolve
                    // them here; all public index keys are canonical revision IDs.
                    result.push((
                        node,
                        GroupReference {
                            entity_id: id,
                            property: name.clone(),
                        },
                        group.id(),
                        group.view.clone(),
                    ));
                }
            }
        }
    }
    result.sort_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
    Ok(result)
}

impl GroupMembershipIndex {
    /// Index references reachable from exactly the selected root rows/edges.
    /// Legacy membership-only references reject rather than reading live values.
    pub fn build(view: GraphView) -> GraphResult<Self> {
        let mut data = GroupData::default();
        let mut pending = Vec::new();
        for (node, reference, group, source) in references(&view)? {
            data.roots.insert(group);
            (if node {
                &mut data.nodes
            } else {
                &mut data.edges
            })
            .entry(group)
            .or_default()
            .push(reference);
            if let std::collections::hash_map::Entry::Vacant(entry) = data.groups.entry(group) {
                entry.insert(source);
                pending.push(group);
            }
        }
        // An iterative traversal visits every shared revision once, including
        // deeply nested groups, without recursive call-stack growth.
        while let Some(parent) = pending.pop() {
            let source = &data.groups[&parent];
            for (_, _, child, source) in references(source)? {
                data.children.entry(parent).or_default().insert(child);
                data.parents.entry(child).or_default().insert(parent);
                if let std::collections::hash_map::Entry::Vacant(entry) = data.groups.entry(child) {
                    entry.insert(source);
                    pending.push(child);
                }
            }
        }
        Ok(Self {
            view,
            data: Arc::new(data),
        })
    }
    /// The root capture and exact selection; nested sources keep their own pins.
    pub fn view(&self) -> &GraphView {
        &self.view
    }
    /// Explicit replacement for a new selection; this index remains unchanged.
    pub fn rebuild(&self, view: GraphView) -> GraphResult<Self> {
        Self::build(view)
    }
    /// Sorted canonical IDs of all reachable revisions, including nested groups.
    pub fn group_ids(&self) -> Vec<SubgraphId> {
        let mut ids: Vec<_> = self.data.groups.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
    /// Sorted revisions directly referenced by selected root properties.
    pub fn root_groups(&self) -> Vec<SubgraphId> {
        self.data.roots.iter().copied().collect()
    }
    /// Exact frozen source selection; not a join with the root graph's node IDs.
    pub fn group_view(&self, group: SubgraphId) -> GraphResult<GraphView> {
        self.data.groups.get(&group).cloned().ok_or_else(|| {
            GraphError::InvalidInput(format!("Group revision {group} is outside this index"))
        })
    }
    fn require(&self, group: SubgraphId) -> GraphResult<()> {
        if self.data.groups.contains_key(&group) {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(format!(
                "Group revision {group} is outside this index"
            )))
        }
    }
    /// Direct nested revisions referenced by the given frozen source.
    pub fn children(&self, group: SubgraphId) -> GraphResult<Vec<SubgraphId>> {
        self.require(group)?;
        Ok(self
            .data
            .children
            .get(&group)
            .into_iter()
            .flatten()
            .copied()
            .collect())
    }
    /// Direct parent revisions within this reachable captured index.
    pub fn parents(&self, group: SubgraphId) -> GraphResult<Vec<SubgraphId>> {
        self.require(group)?;
        Ok(self
            .data
            .parents
            .get(&group)
            .into_iter()
            .flatten()
            .copied()
            .collect())
    }
    /// Membership in this group revision's source-local node namespace.
    pub fn contains_node(&self, group: SubgraphId, node: usize) -> GraphResult<bool> {
        self.require(group)?;
        Ok(self.data.groups[&group].nodes().contains(&node))
    }
    /// Membership of an edge instance, including parallel/loop identities.
    pub fn contains_edge(&self, group: SubgraphId, edge: usize) -> GraphResult<bool> {
        self.require(group)?;
        Ok(self.data.groups[&group].edges().contains(&edge))
    }
    /// Sorted selected node/property references, optionally through nested groups.
    pub fn node_references(
        &self,
        group: SubgraphId,
        recursive: bool,
    ) -> GraphResult<Vec<GroupReference>> {
        self.root_references(group, recursive, true)
    }
    /// Sorted selected edge/property references, optionally through nested groups.
    pub fn edge_references(
        &self,
        group: SubgraphId,
        recursive: bool,
    ) -> GraphResult<Vec<GroupReference>> {
        self.root_references(group, recursive, false)
    }
    fn root_references(
        &self,
        group: SubgraphId,
        recursive: bool,
        node: bool,
    ) -> GraphResult<Vec<GroupReference>> {
        self.require(group)?;
        let postings = if node {
            &self.data.nodes
        } else {
            &self.data.edges
        };
        let mut seen = HashSet::new();
        let mut pending = vec![group];
        let mut rows = BTreeSet::new();
        while let Some(group) = pending.pop() {
            if !seen.insert(group) {
                continue;
            }
            rows.extend(postings.get(&group).into_iter().flatten().cloned());
            if recursive {
                pending.extend(self.data.parents.get(&group).into_iter().flatten().copied());
            }
        }
        Ok(rows.into_iter().collect())
    }
}

impl GraphView {
    /// Explicitly index selected references and reachable frozen group revisions.
    pub fn build_group_index(&self) -> GraphResult<GroupMembershipIndex> {
        GroupMembershipIndex::build(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        state::{GraphSnapshot, PinnedReadState},
        storage::retained_group::RetainedGroup,
        types::GraphType,
    };
    fn view(value: AttrValue, groups: HashMap<usize, RetainedGroup>) -> GraphView {
        let mut snapshot = GraphSnapshot::empty_with_type(0, GraphType::Undirected);
        snapshot.active_nodes = vec![0];
        snapshot
            .node_attributes
            .insert(0, [("reference".into(), value)].into());
        GraphView::all(
            PinnedReadState::from_snapshot(snapshot)
                .unwrap()
                .with_groups(Arc::new(groups)),
        )
    }
    #[test]
    fn local_legacy_aliases_resolve_per_source_without_conflating_revisions() {
        let leaf =
            RetainedGroup::new(view(AttrValue::Int(7), HashMap::new()), "leaf".into()).unwrap();
        let parent_view = view(AttrValue::SubgraphRef(7), [(7, leaf.clone())].into());
        let parent = RetainedGroup::new(parent_view, "parent".into()).unwrap();
        let root = view(AttrValue::SubgraphRef(7), [(7, parent.clone())].into());
        let index = root.build_group_index().unwrap();
        assert_eq!(index.root_groups(), vec![parent.id()]);
        assert_eq!(index.children(parent.id()).unwrap(), vec![leaf.id()]);
        assert_eq!(index.group_ids().len(), 2);
        assert!(index.node_references(leaf.id(), false).unwrap().is_empty());
        assert_eq!(index.node_references(leaf.id(), true).unwrap().len(), 1);
        assert!(index.children(leaf.id()).unwrap().is_empty()); // plain integer 7 is not a group reference
    }
    #[test]
    fn unresolved_legacy_references_reject_but_plain_integer_properties_do_not() {
        assert!(view(AttrValue::SubgraphRef(7), HashMap::new())
            .build_group_index()
            .is_err());
        assert!(view(AttrValue::Int(7), HashMap::new())
            .build_group_index()
            .unwrap()
            .group_ids()
            .is_empty());
    }
}
