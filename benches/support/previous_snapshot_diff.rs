//! Frozen d2abe891 snapshot diff for matched finite-value benchmark control only.
//! Production callers use state::change; this file is never a library module.
use groggy::state::{AttributeChange, EntityType, GraphSnapshot, StateDiff};
use groggy::EdgeEndpointChange;
pub fn previous_diff(before: &GraphSnapshot, other: &GraphSnapshot) -> StateDiff {
    use std::collections::HashSet;

    let self_nodes: HashSet<_> = before.active_nodes.iter().collect();
    let other_nodes: HashSet<_> = other.active_nodes.iter().collect();

    let mut nodes_added: Vec<_> = other
        .active_nodes
        .iter()
        .filter(|&&node| !self_nodes.contains(&node))
        .cloned()
        .collect();

    let mut nodes_removed: Vec<_> = before
        .active_nodes
        .iter()
        .filter(|&&node| !other_nodes.contains(&node))
        .cloned()
        .collect();

    // Compare edges
    let mut edges_added: Vec<_> = other
        .edges
        .iter()
        .filter(|(edge_id, _)| !before.edges.contains_key(edge_id))
        .map(|(edge_id, &(source, target))| (*edge_id, source, target))
        .collect();

    let mut edges_removed: Vec<_> = before
        .edges
        .iter()
        .filter(|(edge_id, _)| !other.edges.contains_key(edge_id))
        .map(|(edge_id, _)| *edge_id)
        .collect();

    let mut edge_endpoint_changes: Vec<_> = other
        .edges
        .iter()
        .filter_map(|(&edge_id, &new_endpoints)| {
            let &old_endpoints = before.edges.get(&edge_id)?;
            (old_endpoints != new_endpoints).then_some(EdgeEndpointChange {
                edge_id,
                old_endpoints,
                new_endpoints,
            })
        })
        .collect();
    edge_endpoint_changes.sort_unstable_by_key(|change| change.edge_id);

    // Compare attributes
    let mut attribute_changes = Vec::new();

    // Node attribute changes
    for &node_id in &other.active_nodes {
        let self_attrs = before
            .node_attributes
            .get(&node_id)
            .cloned()
            .unwrap_or_default();
        let other_attrs = other
            .node_attributes
            .get(&node_id)
            .cloned()
            .unwrap_or_default();

        for (attr_name, new_value) in &other_attrs {
            let old_value = self_attrs.get(attr_name).cloned();
            if old_value.as_ref() != Some(new_value) {
                attribute_changes.push(AttributeChange {
                    entity_type: EntityType::Node,
                    entity_id: node_id as u64,
                    attr_name: attr_name.clone(),
                    old_value,
                    new_value: Some(new_value.clone()),
                });
            }
        }

        // Check for removed attributes
        for (attr_name, old_value) in &self_attrs {
            if !other_attrs.contains_key(attr_name) {
                attribute_changes.push(AttributeChange {
                    entity_type: EntityType::Node,
                    entity_id: node_id as u64,
                    attr_name: attr_name.clone(),
                    old_value: Some(old_value.clone()),
                    new_value: None,
                });
            }
        }
    }

    // Edge attribute changes
    for &edge_id in other.edges.keys() {
        let self_attrs = before
            .edge_attributes
            .get(&edge_id)
            .cloned()
            .unwrap_or_default();
        let other_attrs = other
            .edge_attributes
            .get(&edge_id)
            .cloned()
            .unwrap_or_default();

        for (attr_name, new_value) in &other_attrs {
            let old_value = self_attrs.get(attr_name).cloned();
            if old_value.as_ref() != Some(new_value) {
                attribute_changes.push(AttributeChange {
                    entity_type: EntityType::Edge,
                    entity_id: edge_id as u64,
                    attr_name: attr_name.clone(),
                    old_value,
                    new_value: Some(new_value.clone()),
                });
            }
        }

        // Check for removed attributes
        for (attr_name, old_value) in &self_attrs {
            if !other_attrs.contains_key(attr_name) {
                attribute_changes.push(AttributeChange {
                    entity_type: EntityType::Edge,
                    entity_id: edge_id as u64,
                    attr_name: attr_name.clone(),
                    old_value: Some(old_value.clone()),
                    new_value: None,
                });
            }
        }
    }

    nodes_added.sort_unstable();
    nodes_removed.sort_unstable();
    edges_added.sort_unstable_by_key(|(id, _, _)| *id);
    edges_removed.sort_unstable();
    attribute_changes.sort_unstable_by(|left, right| {
        let kind = |change: &AttributeChange| match change.entity_type {
            EntityType::Node => 0,
            EntityType::Edge => 1,
        };
        (kind(left), left.entity_id, &left.attr_name).cmp(&(
            kind(right),
            right.entity_id,
            &right.attr_name,
        ))
    });

    StateDiff {
        from_state: before.state_id,
        to_state: other.state_id,
        nodes_added,
        nodes_removed,
        edges_added,
        edges_removed,
        edge_endpoint_changes,
        attribute_changes,
    }
}
