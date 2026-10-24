//! Property-only staging for explicitly atomic builder execution.
use crate::{
    api::graph::{kinds, Graph},
    AttrName, AttrValue, BatchEdgeRef, BatchNodeRef, GraphBatch, GraphResult, StateId,
};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug)]
pub(crate) struct PropertyBatch {
    pub expected_state: StateId,
    nodes: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
    edges: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
}
impl PropertyBatch {
    pub fn new(expected_state: StateId) -> Self {
        Self {
            expected_state,
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
        graph.validate_write_head(self.expected_state)?;
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
            self.expected_state,
            "atomic builder properties".into(),
            "groggy".into(),
        )?;
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
        let mut pending = PropertyBatch::new(before);
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
        let mut pending = PropertyBatch::new(before);
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
        let mut empty = PropertyBatch::new(before);
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
        let mut pending = PropertyBatch::new(graph.current_state_id());
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
        let mut pending = PropertyBatch::new(graph.current_state_id());
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
}
