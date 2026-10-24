//! Accept an entirely staged meta-node composition, including its retained group.
use super::*;
use crate::{BatchEdgeRef, BatchNodeRef};

/// Fully evaluated user attributes and boundary edges, without allocated IDs.
pub(crate) struct MetaComposition {
    pub nodes: HashSet<NodeId>,
    pub edges: HashSet<EdgeId>,
    pub kind: String,
    pub attrs: HashMap<AttrName, AttrValue>,
    pub boundary: Vec<(BatchNodeRef, BatchNodeRef, HashMap<AttrName, AttrValue>)>,
    pub remove_originals: bool,
}
impl Graph {
    pub(crate) fn apply_meta_composition(
        &mut self,
        composition: MetaComposition,
    ) -> GraphResult<NodeId> {
        if self.automatic_publication {
            self.validate_write_head(self.current_state_id())?;
        }
        // Validate all user attributes before preparing even a group-storage slot.
        self.validate_new_node_attrs(std::slice::from_ref(&composition.attrs))?;
        if composition.attrs.contains_key("contains_subgraph") {
            return Err(GraphError::InvalidInput(
                "contains_subgraph is managed by meta composition".into(),
            ));
        }
        let stored = self.pool.borrow_mut().prepare_stored_subgraph(
            composition.nodes.clone(),
            composition.edges,
            composition.kind,
        )?;
        let mut batch = GraphBatch {
            add_nodes: 1,
            ..Default::default()
        };
        for (name, value) in composition.attrs {
            batch
                .node_attrs
                .insert(name, vec![(BatchNodeRef::Added(0), value)]);
        }
        batch.node_attrs.insert(
            "contains_subgraph".into(),
            vec![(BatchNodeRef::Added(0), AttrValue::SubgraphRef(stored.id))],
        );
        for (position, (source, target, attrs)) in composition.boundary.into_iter().enumerate() {
            batch.add_edges.push((source, target));
            for (name, value) in attrs {
                batch
                    .edge_attrs
                    .entry(name)
                    .or_default()
                    .push((BatchEdgeRef::Added(position), value));
            }
        }
        if composition.remove_originals {
            let mut nodes: Vec<_> = composition.nodes.into_iter().collect();
            nodes.sort_unstable();
            batch.remove_nodes = nodes.into_iter().map(BatchNodeRef::Existing).collect();
        }
        let counts = self.pool.borrow().statistics();
        let node = counts.node_count;
        if self.automatic_publication {
            let mut prepared = self.prepare_publication(batch, self.current_state_id(), None)?;
            prepared.batch.node_attrs.insert(
                "entity_type".into(),
                vec![(node, AttrValue::Text("meta".into()))],
            );
            self.pool.borrow_mut().install_stored_subgraph(stored);
            self.execute_publication(prepared, "compose_meta_node".into(), "groggy".into())?;
        } else {
            let mut prepared = batch.prepare(
                |id| self.has_node(id),
                |id| self.has_edge(id),
                |id| self.pool.borrow().get_edge_endpoints(id),
                counts.node_count,
                counts.edge_count,
            )?;
            prepared.node_attrs.insert(
                "entity_type".into(),
                vec![(node, AttrValue::Text("meta".into()))],
            );
            self.pool.borrow_mut().install_stored_subgraph(stored);
            self.execute_prepared_mutation(prepared)?;
        }
        Ok(node)
    }
}
