//! Stage a complete table materialization before constructing a destination graph.
use super::{structural, GraphTable, Table};
use crate::{
    api::graph::Graph, AttrValue, BatchEdgeRef, BatchNodeRef, GraphBatch, GraphConfig, GraphError,
    GraphResult,
};
use std::collections::{HashMap, HashSet};

impl GraphTable {
    /// Materialize an undirected publishing graph with fresh IDs in row order.
    /// Unsupported pooled/meta contents reject instead of becoming base entities.
    pub fn to_graph(&self) -> GraphResult<Graph> {
        self.to_graph_with_config(GraphConfig::new())
    }

    /// Materialize one checked state using the supplied destination configuration.
    /// Tables do not retain source direction; callers select it explicitly here.
    pub fn to_graph_with_config(&self, config: GraphConfig) -> GraphResult<Graph> {
        let node_ids = structural::ids(self.nodes().base(), "node_id")?;
        let edge_ids = structural::materialized_edge_ids(self.edges().base())?;
        let nodes = positions(&node_ids, "node_id")?;
        let edges = positions(&edge_ids, "edge_id")?;
        let sources = structural::ids(self.edges().base(), "source")?;
        let targets = structural::ids(self.edges().base(), "target")?;
        let mut batch = GraphBatch {
            add_nodes: nodes.len(),
            ..Default::default()
        };
        let mut pairs = HashSet::new();
        for (row, (source, target)) in sources.into_iter().zip(targets).enumerate() {
            let source = nodes.get(&source).copied().ok_or_else(|| {
                invalid(format!(
                    "source row {row}: node {source} is absent from the table"
                ))
            })?;
            let target = nodes.get(&target).copied().ok_or_else(|| {
                invalid(format!(
                    "target row {row}: node {target} is absent from the table"
                ))
            })?;
            let policy = &self.policy().edge_config;
            if !policy.allow_self_loops && source == target {
                return Err(invalid(format!(
                    "edge row {row}: self-loop disallowed by table policy"
                )));
            }
            let pair =
                if config.graph_type == crate::types::GraphType::Undirected && source > target {
                    (target, source)
                } else {
                    (source, target)
                };
            if !policy.allow_multi_edges && !pairs.insert(pair) {
                return Err(invalid(format!(
                    "edge row {row}: parallel edge disallowed by table policy"
                )));
            }
            batch
                .add_edges
                .push((BatchNodeRef::Added(source), BatchNodeRef::Added(target)));
        }
        for name in self.nodes().base().column_names() {
            if name == "node_id" {
                continue;
            }
            let column = structural::column(self.nodes().base(), name)?;
            for (row, value) in column.iter().enumerate() {
                match name.as_str() {
                    "entity_type" => {
                        if !matches!(value, AttrValue::Null)
                            && value.get_text().as_deref() != Some("base")
                        {
                            return Err(invalid(format!("entity_type row {row}: pooled/meta contents cannot be materialized without their source pool")));
                        }
                    }
                    "contains_subgraph" => {
                        if !matches!(value, AttrValue::Null) {
                            return Err(invalid(format!("contains_subgraph row {row}: requires portable source pool contents")));
                        }
                    }
                    // Projection holes are null cells; labels cannot store Null
                    // in a graph, so this cell means the label property is absent.
                    "labels" if matches!(value, AttrValue::Null) => {}
                    "external_id" => {
                        if let Some(id) = external_id(value, row)? {
                            batch.node_external_ids.push((BatchNodeRef::Added(row), id));
                        }
                    }
                    _ => {
                        batch.node_attrs.entry(name.clone()).or_default().push((
                            BatchNodeRef::Added(row),
                            remap(value, name, row, &nodes, &edges)?,
                        ));
                    }
                }
            }
        }
        for name in self.edges().base().column_names() {
            if matches!(name.as_str(), "edge_id" | "source" | "target") {
                continue;
            }
            let column = structural::column(self.edges().base(), name)?;
            for (row, value) in column.iter().enumerate() {
                if name == "external_id" {
                    if let Some(id) = external_id(value, row)? {
                        batch.edge_external_ids.push((BatchEdgeRef::Added(row), id));
                    }
                } else {
                    batch.edge_attrs.entry(name.clone()).or_default().push((
                        BatchEdgeRef::Added(row),
                        remap(value, name, row, &nodes, &edges)?,
                    ));
                }
            }
        }
        let mut graph = Graph::with_config_manual(config)?;
        graph.set_automatic_publication(true)?;
        if !node_ids.is_empty() {
            graph.apply_batch(batch, 0, "materialize graph table".into(), "groggy".into())?;
        }
        Ok(graph)
    }
}

fn invalid(message: String) -> GraphError {
    GraphError::InvalidInput(message)
}

fn positions(ids: &[usize], name: &str) -> GraphResult<HashMap<usize, usize>> {
    let mut positions = HashMap::with_capacity(ids.len());
    for (row, &id) in ids.iter().enumerate() {
        if positions.insert(id, row).is_some() {
            return Err(invalid(format!(
                "{name} row {row}: duplicate table ID {id}"
            )));
        }
    }
    Ok(positions)
}

fn external_id(value: &AttrValue, row: usize) -> GraphResult<Option<String>> {
    if matches!(value, AttrValue::Null) {
        return Ok(None);
    }
    value.get_text().map(Some).ok_or_else(|| {
        invalid(format!(
            "external_id row {row}: expected text or an unassigned null cell"
        ))
    })
}

fn remap(
    value: &AttrValue,
    name: &str,
    row: usize,
    nodes: &HashMap<usize, usize>,
    edges: &HashMap<usize, usize>,
) -> GraphResult<AttrValue> {
    crate::storage::value_references::remap(value.clone(), nodes, edges, "table")
        .map_err(|error| invalid(format!("{name} row {row}: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        storage::{
            array::BaseArray,
            table::{BaseTable, EdgesTable, NodesTable},
        },
        types::{CompactString, GraphType},
    };

    fn table(
        node_columns: Vec<(&str, Vec<AttrValue>)>,
        edge_columns: Vec<(&str, Vec<AttrValue>)>,
    ) -> GraphTable {
        let base = |columns: Vec<(&str, Vec<AttrValue>)>| {
            BaseTable::from_columns(
                columns
                    .into_iter()
                    .map(|(name, values)| (name.into(), BaseArray::from_attr_values(values)))
                    .collect(),
            )
            .expect("base")
        };
        GraphTable::new(
            NodesTable::from_base_table(base(node_columns)).expect("nodes"),
            EdgesTable::from_base_table(base(edge_columns)).expect("edges"),
        )
    }

    #[test]
    fn row_order_values_identities_and_reference_arrays_materialize_in_one_state() {
        let nodes = NodesTable::new(vec![200, 100])
            .with_attributes(
                "refs".into(),
                HashMap::from([(200, AttrValue::NodeArray(vec![100, 200, 100]))]),
            )
            .expect("refs")
            .with_attributes(
                "external_id".into(),
                HashMap::from([(200, AttrValue::Text("a".into()))]),
            )
            .expect("identity");
        let edges = EdgesTable::new(vec![(900, 100, 200), (800, 200, 100)])
            .with_attributes(
                "refs".into(),
                HashMap::from([(900, AttrValue::EdgeArray(vec![800, 900]))]),
            )
            .expect("refs")
            .with_attributes(
                "w".into(),
                HashMap::from([(900, AttrValue::Float64(1.0000000001))]),
            )
            .expect("weight");
        let source = GraphTable::new(nodes, edges);
        let mut config = GraphConfig::new();
        config.graph_type = GraphType::Directed;
        let graph = source.to_graph_with_config(config).expect("materialize");
        assert!(graph.is_directed() && graph.automatic_publication());
        assert_eq!(graph.current_state_id(), 1);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.edge_endpoints(0).expect("edge"), (1, 0));
        assert_eq!(
            graph.get_node_attr(0, &"refs".into()).expect("refs"),
            Some(AttrValue::NodeArray(vec![1, 0, 1]))
        );
        assert_eq!(
            graph.get_edge_attr(0, &"refs".into()).expect("refs"),
            Some(AttrValue::EdgeArray(vec![1, 0]))
        );
        assert_eq!(
            graph.get_edge_attr(0, &"w".into()).expect("w"),
            Some(AttrValue::Float64(1.0000000001))
        );
        assert_eq!(graph.node_by_external_id("a", None), Some(0));
        assert_eq!(
            source.nodes().node_ids().expect("source IDs"),
            vec![200, 100]
        );
    }

    #[test]
    fn pooled_meta_and_unresolved_references_reject_without_downgrading() {
        for value in [
            AttrValue::SubgraphRef(7),
            AttrValue::NodeArray(vec![99]),
            AttrValue::EdgeArray(vec![99]),
        ] {
            let nodes = NodesTable::new(vec![10])
                .with_attributes("refs".into(), HashMap::from([(10, value)]))
                .expect("refs");
            assert!(GraphTable::new(nodes, EdgesTable::new(vec![]))
                .to_graph()
                .is_err());
        }
        let nodes = NodesTable::new(vec![10])
            .with_attributes(
                "entity_type".into(),
                HashMap::from([(10, AttrValue::CompactText(CompactString::new("meta")))]),
            )
            .expect("meta");
        assert!(GraphTable::new(nodes, EdgesTable::new(vec![]))
            .to_graph()
            .is_err());
    }

    #[test]
    fn null_edge_ids_are_assigned_locally_and_overflow_is_checked() {
        let source = table(
            vec![("node_id", vec![AttrValue::Int(10)])],
            vec![
                ("edge_id", vec![AttrValue::Int(50), AttrValue::Null]),
                ("source", vec![AttrValue::Int(10); 2]),
                ("target", vec![AttrValue::Int(10); 2]),
                (
                    "refs",
                    vec![AttrValue::EdgeArray(vec![51]), AttrValue::Null],
                ),
            ],
        );
        let graph = source.to_graph().expect("null compatibility");
        assert_eq!(
            graph.get_edge_attr(0, &"refs".into()).expect("refs"),
            Some(AttrValue::EdgeArray(vec![1]))
        );
        assert_eq!(
            source.edges().base().column("edge_id").expect("IDs").get(1),
            Some(&AttrValue::Null)
        );
        let overflow = table(
            vec![("node_id", vec![AttrValue::Int(10)])],
            vec![
                ("edge_id", vec![AttrValue::Int(i64::MAX), AttrValue::Null]),
                ("source", vec![AttrValue::Int(10); 2]),
                ("target", vec![AttrValue::Int(10); 2]),
            ],
        );
        assert!(overflow
            .to_graph()
            .expect_err("overflow")
            .to_string()
            .contains("integer range"));
    }

    #[test]
    fn structural_rows_and_configured_loop_parallel_policies_are_checked() {
        for value in [
            AttrValue::Null,
            AttrValue::Int(-1),
            AttrValue::Float64(10.0),
            AttrValue::Text("10".into()),
        ] {
            let source = table(
                vec![("node_id", vec![AttrValue::Int(10)])],
                vec![
                    ("edge_id", vec![AttrValue::Int(3)]),
                    ("source", vec![value]),
                    ("target", vec![AttrValue::Int(10)]),
                ],
            );
            assert!(source.to_graph().is_err());
        }
        let mut source = GraphTable::new(
            NodesTable::new(vec![10, 20]),
            EdgesTable::new(vec![(1, 10, 20), (2, 20, 10)]),
        );
        let mut policy = source.policy().clone();
        policy.edge_config.allow_multi_edges = false;
        source.set_policy(policy);
        assert!(source.to_graph().is_err());
        let mut config = GraphConfig::new();
        config.graph_type = GraphType::Directed;
        assert!(source.to_graph_with_config(config).is_ok());
        let mut loops = GraphTable::new(
            NodesTable::new(vec![10]),
            EdgesTable::new(vec![(1, 10, 10)]),
        );
        let mut policy = loops.policy().clone();
        policy.edge_config.allow_self_loops = false;
        loops.set_policy(policy);
        assert!(loops.to_graph().is_err());
    }

    #[test]
    fn compact_ids_and_empty_graphs_have_checked_publishing_defaults() {
        let source = table(
            vec![("node_id", vec![AttrValue::SmallInt(20)])],
            vec![
                ("edge_id", vec![AttrValue::SmallInt(4)]),
                ("source", vec![AttrValue::SmallInt(20)]),
                ("target", vec![AttrValue::SmallInt(20)]),
            ],
        );
        assert_eq!(source.to_graph().expect("compact").edge_count(), 1);
        let empty = GraphTable::empty().to_graph().expect("empty");
        assert!(empty.automatic_publication());
        assert!(!empty.has_uncommitted_changes());
        assert_eq!(empty.current_state_id(), 0);
    }
}
