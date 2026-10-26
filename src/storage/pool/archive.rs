//! Portable raw column positions and retained source closure; caches are rebuilt.
use super::*;
use crate::storage::retained_group::{self, GroupDocument};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PoolDocument {
    graph_type: GraphType,
    node_attributes: BTreeMap<String, Vec<Value>>,
    edge_attributes: BTreeMap<String, Vec<Value>>,
    edge_sources: Vec<NodeId>,
    edge_targets: Vec<NodeId>,
    edge_alive: Vec<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sparse_edges: Option<Vec<(EdgeId, NodeId, NodeId)>>,
    stored_subgraphs: BTreeMap<SubgraphId, (HashSet<NodeId>, HashSet<EdgeId>, String)>,
    groups: Vec<(SubgraphId, GroupDocument)>,
    group_aliases: BTreeMap<SubgraphId, SubgraphId>,
    #[serde(default, skip_serializing_if = "PurgedIds::is_empty")]
    purged_nodes: PurgedIds,
    #[serde(default, skip_serializing_if = "PurgedIds::is_empty")]
    purged_edges: PurgedIds,
    next_node_id: NodeId,
    next_edge_id: EdgeId,
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(format!("Invalid graph archive pool: {message}"))
}
impl GraphPool {
    pub(crate) fn archive_document(&self) -> GraphResult<PoolDocument> {
        let encode = |columns: &HashMap<String, AttributeColumn>, kind: &str| {
            columns
                .iter()
                .map(|(name, column)| {
                    Ok((
                        name.clone(),
                        column
                            .values
                            .iter()
                            .enumerate()
                            .map(|(index, value)| {
                                retained_group::encode(value).map_err(|error| {
                                    invalid(&format!(
                                        "{kind} column {name:?} value {index}: {error}"
                                    ))
                                })
                            })
                            .collect::<GraphResult<_>>()?,
                    ))
                })
                .collect::<GraphResult<_>>()
        };
        let mut groups = Vec::new();
        let mut seen = HashSet::new();
        let mut roots: Vec<_> = self.retained_groups.iter().collect();
        roots.sort_unstable_by_key(|(id, _)| **id);
        for (_, group) in roots {
            for (id, document) in retained_group::documents(group)? {
                if seen.insert(id) {
                    groups.push((id, document));
                }
            }
        }
        Ok(PoolDocument {
            graph_type: self.graph_type,
            node_attributes: encode(&self.node_attributes, "node")?,
            edge_attributes: encode(&self.edge_attributes, "edge")?,
            edge_sources: self.edge_sources.clone(),
            edge_targets: self.edge_targets.clone(),
            edge_alive: self.edge_alive.clone(),
            sparse_edges: self.sparse_edges.as_ref().map(|edges| {
                let mut records: Vec<_> = edges.iter().map(|(&id, &(u, v))| (id, u, v)).collect();
                records.sort_unstable_by_key(|&(id, _, _)| id);
                records
            }),
            stored_subgraphs: self
                .stored_subgraphs
                .iter()
                .map(|(&id, v)| (id, v.clone()))
                .collect(),
            groups,
            group_aliases: self
                .retained_groups
                .iter()
                .map(|(&id, group)| (id, group.id()))
                .collect(),
            purged_nodes: self.purged_nodes.clone(),
            purged_edges: self.purged_edges.clone(),
            next_node_id: self.next_node_id,
            next_edge_id: self.next_edge_id,
        })
    }
}
impl PoolDocument {
    #[cfg(test)]
    pub(crate) fn set_test_purged_nodes(&mut self, ranges: Vec<(usize, usize)>) {
        self.purged_nodes = PurgedIds(ranges);
    }

    pub(crate) fn replace_compacted_columns(
        &mut self,
        nodes: &super::compaction::Columns,
        edges: &super::compaction::Columns,
    ) -> GraphResult<()> {
        let encode =
            |columns: &super::compaction::Columns| -> GraphResult<BTreeMap<String, Vec<Value>>> {
                columns
                    .iter()
                    .map(|(name, values)| {
                        Ok((
                            name.clone(),
                            values
                                .iter()
                                .map(retained_group::encode)
                                .collect::<GraphResult<_>>()?,
                        ))
                    })
                    .collect()
            };
        let nodes = encode(nodes)?;
        let edges = encode(edges)?;
        self.node_attributes = nodes;
        self.edge_attributes = edges;
        Ok(())
    }

    pub(crate) fn restore(self, graph_type: GraphType) -> GraphResult<GraphPool> {
        if self.graph_type != graph_type
            || self.edge_sources.len() != self.edge_targets.len()
            || self.edge_sources.len() != self.edge_alive.len()
            || self.edge_sources.len() > self.next_edge_id
            || self.next_node_id == usize::MAX
            || self.next_edge_id == usize::MAX
        {
            return Err(invalid("inconsistent topology or allocation counters"));
        }
        self.purged_nodes.validate(self.next_node_id)?;
        self.purged_edges.validate(self.next_edge_id)?;
        if let Some(edges) = &self.sparse_edges {
            if !self.edge_sources.is_empty()
                || !self.edge_targets.is_empty()
                || !self.edge_alive.is_empty()
            {
                return Err(invalid("mixed sparse and dense endpoint storage"));
            }
            let mut previous = None;
            for &(id, u, v) in edges {
                if previous.is_some_and(|p| id <= p)
                    || id >= self.next_edge_id
                    || self.purged_edges.contains(id)
                    || u >= self.next_node_id
                    || v >= self.next_node_id
                    || self.purged_nodes.contains(u)
                    || self.purged_nodes.contains(v)
                {
                    return Err(invalid("invalid sparse edge record"));
                }
                previous = Some(id);
            }
        }
        for (id, (&u, &v)) in self.edge_sources.iter().zip(&self.edge_targets).enumerate() {
            if self.edge_alive[id]
                && (u >= self.next_node_id
                    || v >= self.next_node_id
                    || self.purged_nodes.contains(u)
                    || self.purged_nodes.contains(v)
                    || self.purged_edges.contains(id))
            {
                return Err(invalid("edge endpoints exceed node allocation"));
            }
        }
        for (id, (nodes, edges, _)) in &self.stored_subgraphs {
            // Frozen sources have their own ID domain, independent of this pool.
            if self.group_aliases.contains_key(id) {
                continue;
            }
            if nodes
                .iter()
                .any(|id| *id >= self.next_node_id || self.purged_nodes.contains(*id))
                || edges
                    .iter()
                    .any(|id| *id >= self.next_edge_id || self.purged_edges.contains(*id))
            {
                return Err(invalid("stored membership exceeds entity allocation"));
            }
        }
        let decode = |columns: BTreeMap<String, Vec<Value>>| {
            columns
                .into_iter()
                .map(|(name, values)| {
                    let mut column = AttributeColumn::new();
                    column.values = values
                        .into_iter()
                        .map(retained_group::decode)
                        .collect::<GraphResult<_>>()?;
                    Ok((name, column))
                })
                .collect::<GraphResult<_>>()
        };
        let mut pool = GraphPool::new_with_type(graph_type);
        pool.purged_nodes = self.purged_nodes;
        pool.purged_edges = self.purged_edges;
        pool.next_node_id = self.next_node_id;
        pool.next_edge_id = self.next_edge_id;
        pool.edge_sources = self.edge_sources;
        pool.edge_targets = self.edge_targets;
        pool.edge_alive = self.edge_alive;
        pool.sparse_edges = self
            .sparse_edges
            .map(|edges| edges.into_iter().map(|(id, u, v)| (id, (u, v))).collect());
        pool.node_attributes = decode(self.node_attributes)?;
        pool.edge_attributes = decode(self.edge_attributes)?;
        // Restore frozen dependencies before legacy memberships, which may alias them.
        let prepared = pool.prepare_group_documents(self.groups)?;
        for group in prepared {
            pool.install_stored_subgraph(group);
        }
        let mut registry = HashMap::new();
        for (alias, canonical) in self.group_aliases {
            let group = pool
                .retained_groups
                .get(&canonical)
                .ok_or_else(|| invalid("missing frozen group alias source"))?;
            let expected = (
                group.view.nodes().clone(),
                group.view.edges().clone(),
                group.kind.clone(),
            );
            if self.stored_subgraphs.get(&alias) != Some(&expected) {
                return Err(invalid("stored membership conflicts with frozen source"));
            }
            registry.insert(alias, group.clone());
        }
        pool.retained_groups = std::sync::Arc::new(registry);
        pool.stored_subgraphs = self.stored_subgraphs.into_iter().collect();
        for value in pool
            .node_attributes
            .values()
            .chain(pool.edge_attributes.values())
            .flat_map(|c| &c.values)
        {
            pool.validate_archive_value(value)?;
        }
        Ok(pool)
    }
}
impl GraphPool {
    pub(crate) fn validate_archive_value(&self, value: &AttrValue) -> GraphResult<()> {
        let invalid_reference = match value {
            AttrValue::SubgraphRef(id) => !self.stored_subgraphs.contains_key(id),
            _ => false,
        };
        if invalid_reference {
            return Err(invalid("dangling typed entity/group reference"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod purge_tests {
    use super::*;
    #[test]
    fn sparse_endpoint_archives_reject_duplicate_unsorted_mixed_and_invalid_records() {
        let mut pool = GraphPool::new();
        pool.ensure_node_id_exists(2).unwrap();
        pool.add_edges(&[(0, 1), (1, 2), (0, 2)]);
        let mut edges = HashSet::from([1, 2]);
        pool.reclaim_storage(HashSet::from([0, 1, 2]), &mut edges, &HashSet::new())
            .unwrap();
        let valid = pool.archive_document().unwrap();
        assert!(valid.clone().restore(pool.graph_type).is_ok());
        for records in [
            vec![(1, 1, 2), (1, 1, 2)],
            vec![(2, 0, 2), (1, 1, 2)],
            vec![(3, 0, 2)],
            vec![(0, 0, 1)],
            vec![(1, 3, 2)],
        ] {
            let mut doc = valid.clone();
            doc.sparse_edges = Some(records);
            assert!(doc.restore(pool.graph_type).is_err());
        }
        let mut doc = valid.clone();
        doc.edge_sources.push(0);
        doc.edge_targets.push(1);
        doc.edge_alive.push(true);
        assert!(doc.restore(pool.graph_type).is_err());
        let mut doc = valid;
        doc.purged_nodes = PurgedIds(vec![(2, 3)]);
        assert!(doc.restore(pool.graph_type).is_err());
    }
}
