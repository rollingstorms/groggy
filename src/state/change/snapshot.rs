//! One deterministic logical snapshot-to-change conversion.
use crate::state::GraphSnapshot;
use crate::GraphResult;
use std::collections::HashSet;

pub(crate) fn snapshot_changes(
    before: &GraphSnapshot,
    after: &GraphSnapshot,
) -> GraphResult<crate::state::ChangeSet> {
    snapshot_changes_by(before, after, super::same_value)
}
pub(crate) fn snapshot_changes_by<E>(
    before: &GraphSnapshot,
    after: &GraphSnapshot,
    mut same: impl FnMut(&crate::AttrValue, &crate::AttrValue) -> Result<bool, E>,
) -> Result<crate::state::ChangeSet, E> {
    use std::collections::BTreeSet;
    let mut changes = crate::state::ChangeSet::new();
    let old_nodes: HashSet<_> = before.active_nodes.iter().copied().collect();
    let new_nodes: HashSet<_> = after.active_nodes.iter().copied().collect();
    changes.nodes_added = new_nodes.difference(&old_nodes).copied().collect();
    changes.nodes_removed = old_nodes.difference(&new_nodes).copied().collect();
    changes.nodes_added.sort_unstable();
    changes.nodes_removed.sort_unstable();
    for (&id, &(u, v)) in &after.edges {
        match before.edges.get(&id) {
            None => changes.edges_added.push((id, u, v)),
            Some(&old) if old != (u, v) => {
                changes
                    .edge_endpoint_changes
                    .push(crate::EdgeEndpointChange {
                        edge_id: id,
                        old_endpoints: old,
                        new_endpoints: (u, v),
                    })
            }
            _ => {}
        }
    }
    changes.edges_added.sort_unstable();
    changes.edges_removed = before
        .edges
        .keys()
        .filter(|id| !after.edges.contains_key(id))
        .copied()
        .collect();
    changes.edges_removed.sort_unstable();
    changes
        .edge_endpoint_changes
        .sort_unstable_by_key(|c| c.edge_id);
    for (nodes, old_rows, new_rows, output) in [
        (
            true,
            &before.node_attributes,
            &after.node_attributes,
            &mut changes.node_attr_changes,
        ),
        (
            false,
            &before.edge_attributes,
            &after.edge_attributes,
            &mut changes.edge_attr_changes,
        ),
    ] {
        if old_rows.is_empty() && new_rows.is_empty() {
            continue;
        }
        let ids: Vec<_> = if nodes {
            new_nodes.iter().copied().collect()
        } else {
            after.edges.keys().copied().collect()
        };
        for id in ids {
            let was_active = if nodes {
                old_nodes.contains(&id)
            } else {
                before.edges.contains_key(&id)
            };
            let old = was_active.then(|| old_rows.get(&id)).flatten();
            let new = new_rows.get(&id);
            let names: BTreeSet<_> = old
                .into_iter()
                .flat_map(|row| row.keys())
                .chain(new.into_iter().flat_map(|row| row.keys()))
                .collect();
            for name in names {
                let a = old.and_then(|r| r.get(name));
                let b = new.and_then(|r| r.get(name));
                let equal = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => same(a, b)?,
                    _ => false,
                };
                if !equal {
                    output.push((id, name.clone(), a.cloned(), b.cloned()));
                }
            }
        }
    }
    changes
        .node_attr_changes
        .sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    changes
        .edge_attr_changes
        .sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    Ok(changes)
}
