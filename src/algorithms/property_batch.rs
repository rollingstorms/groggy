//! Property-only staging for explicitly atomic builder execution.
use super::Context;
use crate::{
    api::graph::WriteCoordinate,
    api::graph::{kinds, Graph},
    AttrName, AttrValue, BatchEdgeRef, BatchNodeRef, EdgeId, GraphBatch, GraphResult, NodeId,
};
use crate::{subgraphs::Subgraph, traits::SubgraphOperations};
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug)]
pub(crate) struct PropertyBatch {
    coordinate: WriteCoordinate,
    nodes: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
    edges: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
}
impl PropertyBatch {
    pub fn new(coordinate: WriteCoordinate) -> Self {
        Self {
            coordinate,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
        }
    }
    pub fn get(&self, node: bool, id: usize, name: &AttrName) -> Option<&AttrValue> {
        (if node { &self.nodes } else { &self.edges })
            .get(name)?
            .get(&id)
    }
    pub fn stage(
        &mut self,
        node: bool,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
    ) -> GraphResult<()> {
        // Caller validates the entire input before any staging. Normalization is
        // identical to ordinary graph writes, including empty labels vectors.
        let mut attrs = attrs;
        for (name, rows) in &mut attrs {
            for (_, value) in rows {
                kinds::normalize_kind_property_in_place(name, value, node)?;
            }
        }
        let columns = if node {
            &mut self.nodes
        } else {
            &mut self.edges
        };
        for (name, rows) in attrs {
            if !rows.is_empty() {
                columns.entry(name).or_default().extend(rows);
            }
        }
        Ok(())
    }
    pub fn accept(self, graph: &mut Graph) -> GraphResult<()> {
        graph.validate_write_coordinate(&self.coordinate)?;
        if self.nodes.is_empty() && self.edges.is_empty() {
            return Ok(());
        }
        let batch = GraphBatch {
            node_attrs: self
                .nodes
                .into_iter()
                .map(|(name, rows)| {
                    (
                        name,
                        rows.into_iter()
                            .map(|(id, value)| (BatchNodeRef::Existing(id), value))
                            .collect(),
                    )
                })
                .collect(),
            edge_attrs: self
                .edges
                .into_iter()
                .map(|(name, rows)| {
                    (
                        name,
                        rows.into_iter()
                            .map(|(id, value)| (BatchEdgeRef::Existing(id), value))
                            .collect(),
                    )
                })
                .collect(),
            ..Default::default()
        };
        graph.apply_batch(
            batch,
            self.coordinate.state_id(),
            "atomic builder properties".into(),
            "groggy".into(),
        )?;
        Ok(())
    }
}

impl Context {
    /// Read a node property, including prior accepted attachments in this run.
    pub(crate) fn node_attribute(
        &self,
        subgraph: &Subgraph,
        id: NodeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        if let Some(value) = self
            .property_batch
            .as_ref()
            .and_then(|batch| batch.get(true, id, name))
        {
            return Ok(Some(value.clone()));
        }
        subgraph.get_node_attribute(id, name)
    }
    /// Read an edge property through the same property-only overlay.
    pub(crate) fn edge_attribute(
        &self,
        subgraph: &Subgraph,
        id: EdgeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        if let Some(value) = self
            .property_batch
            .as_ref()
            .and_then(|batch| batch.get(false, id, name))
        {
            return Ok(Some(value.clone()));
        }
        subgraph.get_edge_attribute(id, name)
    }

    /// Publish a property column or return an owned output according to Context.
    /// Transient columns still use canonical target/kind validation.
    pub(crate) fn attach_node_column(
        &mut self,
        subgraph: &Subgraph,
        name: AttrName,
        values: Vec<(NodeId, AttrValue)>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("node attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph
                .set_node_attr_column(name, values)
                .map_err(|err| anyhow!(err));
        }
        self.attach_node_attrs(subgraph, HashMap::from([(name, values)]))
    }

    /// Attach multiple node columns through one validated persistence boundary.
    pub(crate) fn attach_node_attrs(
        &mut self,
        subgraph: &Subgraph,
        attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("node attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph.set_node_attrs(attrs).map_err(|err| anyhow!(err));
        }
        subgraph
            .graph()
            .borrow()
            .validate_node_attrs(&attrs, None)?;
        if let Some(batch) = self.property_batch.as_mut() {
            return batch.stage(true, attrs).map_err(|err| anyhow!(err));
        }
        for (name, mut values) in attrs {
            values.sort_unstable_by_key(|(id, _)| *id);
            self.add_output(
                format!("node_attributes.{name}"),
                super::AlgorithmOutput::NodeAttributes(values),
            );
        }
        Ok(())
    }

    /// Attach edge columns or expose owned results without changing graph state.
    pub(crate) fn attach_edge_attrs(
        &mut self,
        subgraph: &Subgraph,
        attrs: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("edge attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph.set_edge_attrs(attrs).map_err(|err| anyhow!(err));
        }
        subgraph
            .graph()
            .borrow()
            .validate_edge_attrs(&attrs, None)?;
        if let Some(batch) = self.property_batch.as_mut() {
            return batch.stage(false, attrs).map_err(|err| anyhow!(err));
        }
        for (name, mut values) in attrs {
            values.sort_unstable_by_key(|(id, _)| *id);
            self.add_output(
                format!("edge_attributes.{name}"),
                super::AlgorithmOutput::EdgeAttributes(values),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_values_are_one_mixed_batch_and_prior_graph_values_stay_unchanged_until_accept() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(2).expect("nodes");
        graph.add_edge(0, 1).expect("edge");
        let before = graph.current_state_id();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        pending
            .stage(
                true,
                HashMap::from([(
                    "x".into(),
                    vec![(0, AttrValue::Int(2)), (1, AttrValue::Null)],
                )]),
            )
            .expect("overwrite");
        pending
            .stage(
                false,
                HashMap::from([("w".into(), vec![(0, AttrValue::Float64(1.0000000001))])]),
            )
            .expect("edge");
        assert_eq!(pending.get(true, 0, &"x".into()), Some(&AttrValue::Int(2)));
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("old x"), None);
        pending.accept(&mut graph).expect("accept");
        assert_eq!(graph.current_state_id(), before + 1);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(
            graph.get_node_attr(0, &"x".into()).expect("x"),
            Some(AttrValue::Int(2))
        );
        assert_eq!(
            graph.get_edge_attr(0, &"w".into()).expect("w"),
            Some(AttrValue::Float64(1.0000000001))
        );
    }
    #[test]
    fn normalization_is_visible_to_later_reads_and_empty_columns_accept_nothing() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(1).expect("node");
        let before = graph.current_state_id();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([(
                    "labels".into(),
                    vec![(0, AttrValue::Float64Vec(Vec::new()))],
                )]),
            )
            .expect("normalize");
        assert_eq!(
            pending.get(true, 0, &"labels".into()),
            Some(&AttrValue::TextVec(Vec::new()))
        );
        let mut empty = PropertyBatch::new(graph.write_coordinate());
        empty
            .stage(true, HashMap::from([("x".into(), Vec::new())]))
            .expect("empty");
        empty.accept(&mut graph).expect("no-op");
        assert_eq!(graph.current_state_id(), before);
    }
    #[test]
    fn stale_or_dirty_heads_reject_without_accepting_staged_values() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(1).expect("node");
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        graph
            .set_node_attr(0, "other".into(), AttrValue::Int(2))
            .expect("external write");
        assert!(pending.accept(&mut graph).is_err());
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
        graph.set_automatic_publication(false).expect("manual");
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        graph
            .set_node_attr(0, "other".into(), AttrValue::Int(3))
            .expect("dirty write");
        assert!(pending.accept(&mut graph).is_err());
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
    }

    #[test]
    fn branch_or_owner_changes_reject_even_when_the_head_and_ids_match() {
        for foreign_owner in [false, true] {
            let mut graph = Graph::new_publishing();
            graph.insert_nodes(1).expect("node");
            let head = graph.current_state_id();
            let mut pending = PropertyBatch::new(graph.write_coordinate());
            pending
                .stage(
                    true,
                    HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
                )
                .expect("stage");
            if foreign_owner {
                graph = Graph::new_publishing();
                graph.insert_nodes(1).expect("same foreign head");
            } else {
                graph.create_branch("other".into()).expect("branch");
                graph.checkout_branch("other".into()).expect("switch");
            }
            assert_eq!(graph.current_state_id(), head);
            let error = pending.accept(&mut graph).expect_err("coordinate rejects");
            assert!(error.to_string().contains("coordinate"));
            assert_eq!(graph.current_state_id(), head);
            assert!(!graph.has_uncommitted_changes());
            assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
            assert_eq!(graph.add_node().expect("no consumed IDs"), 1);
        }
    }
}
