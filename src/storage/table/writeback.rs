//! Explicit, columnar table property updates through the checked graph coordinator.
use super::{structural::ids, BaseTable, GraphTable, Table};
use crate::{
    api::graph::Graph, AttrName, AttrValue, BatchEdgeRef, BatchNodeRef, BatchResult, EdgeId,
    GraphBatch, GraphError, GraphResult, NodeId, StateId, TimestampMicros,
};
use std::collections::{HashMap, HashSet};

/// Explicit identity and column selection for a mixed table property update.
/// Mappings cover every structural row and must be one-to-one. They assert target
/// identity; table row IDs are never assumed to belong to the destination graph.
#[derive(Clone, Debug, Default)]
pub struct TableWriteback {
    /// Table node ID to destination node ID, including endpoint-only rows.
    pub node_mapping: HashMap<NodeId, NodeId>,
    /// Table edge ID to destination edge ID. Parallel edges keep distinct mappings.
    pub edge_mapping: HashMap<EdgeId, EdgeId>,
    /// Selected property columns; no implicit writes of structural/identity data.
    pub node_columns: Vec<AttrName>,
    /// Selected edge property columns; omitted columns are unchanged.
    pub edge_columns: Vec<AttrName>,
    /// None uses the acceptance clock, shared by the entire update.
    pub effective_at: Option<TimestampMicros>,
}

impl GraphTable {
    /// Apply selected property columns as one checked mutation of existing entities.
    /// Structural/system columns cannot be changed. Endpoints must agree with the
    /// explicit mappings. Null cells assign Null; omitted columns are unchanged.
    /// A stale/dirty graph or any invalid row rejects before mutation. An empty
    /// column selection validates identity but publishes no version.
    pub fn update_graph(
        &self,
        graph: &mut Graph,
        expected_state: StateId,
        update: &TableWriteback,
    ) -> GraphResult<BatchResult> {
        graph.validate_write_head(expected_state)?;
        let nodes = ids(self.nodes().base(), "node_id")?;
        let edges = ids(self.edges().base(), "edge_id")?;
        let sources = ids(self.edges().base(), "source")?;
        let targets = ids(self.edges().base(), "target")?;
        validate_mapping(&nodes, &update.node_mapping, "node", |id| {
            graph.contains_node(id)
        })?;
        validate_mapping(&edges, &update.edge_mapping, "edge", |id| {
            graph.contains_edge(id)
        })?;
        for (row, &edge) in edges.iter().enumerate() {
            let source = update.node_mapping.get(&sources[row]);
            let target = update.node_mapping.get(&targets[row]);
            let Some((&source, &target)) = source.zip(target) else {
                return Err(invalid(format!(
                    "edge row {row}: endpoints require mapped node rows"
                )));
            };
            let actual = graph.edge_endpoints(update.edge_mapping[&edge])?;
            if actual != (source, target) && (graph.is_directed() || actual != (target, source)) {
                return Err(invalid(format!(
                    "edge row {row}: mapped endpoints differ from destination topology"
                )));
            }
        }
        let node_columns = columns(
            self.nodes().base(),
            &update.node_columns,
            &["node_id", "entity_type", "contains_subgraph", "external_id"],
        )?;
        let edge_columns = columns(
            self.edges().base(),
            &update.edge_columns,
            &["edge_id", "source", "target", "external_id"],
        )?;
        let mut batch = GraphBatch {
            effective_at: update.effective_at,
            ..Default::default()
        };
        for (name, values) in node_columns {
            batch.node_attrs.insert(
                name,
                nodes
                    .iter()
                    .zip(values)
                    .map(|(id, value)| (BatchNodeRef::Existing(update.node_mapping[id]), value))
                    .collect(),
            );
        }
        for (name, values) in edge_columns {
            batch.edge_attrs.insert(
                name,
                edges
                    .iter()
                    .zip(values)
                    .map(|(id, value)| (BatchEdgeRef::Existing(update.edge_mapping[id]), value))
                    .collect(),
            );
        }
        if batch.node_attrs.values().all(Vec::is_empty)
            && batch.edge_attrs.values().all(Vec::is_empty)
        {
            return Ok(BatchResult {
                state_id: expected_state,
                added_nodes: Vec::new(),
                added_edges: Vec::new(),
            });
        }
        graph.apply_batch(
            batch,
            expected_state,
            "table_update".into(),
            "groggy".into(),
        )
    }
}

fn invalid(message: String) -> GraphError {
    GraphError::InvalidInput(message)
}

fn validate_mapping(
    rows: &[usize],
    mapping: &HashMap<usize, usize>,
    kind: &str,
    active: impl Fn(usize) -> bool,
) -> GraphResult<()> {
    let mut seen = HashSet::new();
    let mut destinations = HashSet::new();
    for (row, id) in rows.iter().enumerate() {
        if !seen.insert(id) {
            return Err(invalid(format!(
                "{kind} row {row}: duplicate table ID {id}; reduce joined rows explicitly"
            )));
        }
        let target = mapping.get(id).ok_or_else(|| {
            invalid(format!(
                "{kind} row {row}: missing explicit mapping for {id}"
            ))
        })?;
        if !destinations.insert(target) {
            return Err(invalid(format!(
                "{kind} row {row}: ambiguous duplicate destination {target}; reduce explicitly"
            )));
        }
        if !active(*target) {
            return Err(invalid(format!(
                "{kind} row {row}: destination {target} is inactive"
            )));
        }
    }
    if mapping.len() != rows.len() {
        return Err(invalid(format!(
            "{kind} mapping contains IDs absent from the table"
        )));
    }
    Ok(())
}

fn columns(
    table: &BaseTable,
    names: &[AttrName],
    reserved: &[&str],
) -> GraphResult<Vec<(AttrName, Vec<AttrValue>)>> {
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(names.len());
    for name in names {
        if reserved.contains(&name.as_str()) {
            return Err(invalid(format!(
                "{name}: structural or system-managed column cannot be written back"
            )));
        }
        if !seen.insert(name) {
            return Err(invalid(format!("Duplicate writeback column {name}")));
        }
        let column = table
            .column(name)
            .ok_or_else(|| invalid(format!("Missing writeback column {name}")))?;
        if column.len() != table.nrows() {
            return Err(invalid(format!("{name}: inconsistent column length")));
        }
        let mut values = Vec::with_capacity(column.len());
        for (row, value) in column.iter().enumerate() {
            if matches!(
                value,
                AttrValue::SubgraphRef(_) | AttrValue::NodeArray(_) | AttrValue::EdgeArray(_)
            ) {
                return Err(invalid(format!(
                    "{name} row {row}: graph-local references require explicit reference remapping"
                )));
            }
            values.push(value.clone());
        }
        result.push((name.clone(), values));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        array::BaseArray,
        table::{EdgesTable, NodesTable},
    };

    fn fixture() -> (Graph, GraphTable, TableWriteback) {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(2).expect("nodes");
        graph.add_edge(0, 1).expect("edge");
        let nodes = NodesTable::new(vec![20, 10])
            .with_attributes(
                "x".into(),
                HashMap::from([(20, AttrValue::Int(4)), (10, AttrValue::Null)]),
            )
            .expect("node column");
        let edges = EdgesTable::new(vec![(30, 10, 20)])
            .with_attributes(
                "w".into(),
                HashMap::from([(30, AttrValue::Float64(1.0000000001))]),
            )
            .expect("edge column");
        let update = TableWriteback {
            node_mapping: HashMap::from([(10, 0), (20, 1)]),
            edge_mapping: HashMap::from([(30, 0)]),
            node_columns: vec!["x".into()],
            edge_columns: vec!["w".into()],
            effective_at: Some(42),
        };
        (graph, GraphTable::new(nodes, edges), update)
    }

    #[test]
    fn mixed_columns_use_explicit_row_mapping_and_one_accepted_state() {
        let (mut graph, table, update) = fixture();
        let before = graph.current_state_id();
        let result = table
            .update_graph(&mut graph, before, &update)
            .expect("writeback");
        assert_eq!(result.state_id, before + 1);
        assert!(result.added_nodes.is_empty() && result.added_edges.is_empty());
        assert_eq!(
            graph.get_node_attr(1, &"x".into()).expect("x"),
            Some(AttrValue::Int(4))
        );
        assert_eq!(
            graph.get_node_attr(0, &"x".into()).expect("null"),
            Some(AttrValue::Null)
        );
        assert_eq!(
            graph.get_edge_attr(0, &"w".into()).expect("w"),
            Some(AttrValue::Float64(1.0000000001))
        );
        assert!(!graph.has_uncommitted_changes());
    }

    #[test]
    fn late_property_rejection_preserves_all_rows_versions_and_ids() {
        let (mut graph, table, mut update) = fixture();
        let state = graph.current_state_id();
        update.edge_columns.push("missing".into());
        assert!(table.update_graph(&mut graph, state, &update).is_err());
        assert_eq!(graph.get_node_attr(1, &"x".into()).expect("x"), None);
        assert_eq!(graph.current_state_id(), state);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.try_add_node().expect("node"), 2);
        assert_eq!(graph.add_edge(0, 1).expect("edge"), 1);
    }

    #[test]
    fn joined_duplicate_ids_and_graph_local_references_reject() {
        let (mut graph, table, update) = fixture();
        let state = graph.current_state_id();
        let duplicated = GraphTable::new(NodesTable::new(vec![10, 10]), table.edges().clone());
        assert!(duplicated.update_graph(&mut graph, state, &update).is_err());
        for value in [
            AttrValue::SubgraphRef(0),
            AttrValue::NodeArray(vec![0]),
            AttrValue::EdgeArray(vec![0]),
        ] {
            let nodes = NodesTable::new(vec![20, 10])
                .with_attributes("x".into(), HashMap::from([(20, value)]))
                .expect("property");
            assert!(GraphTable::new(nodes, table.edges().clone())
                .update_graph(&mut graph, state, &update)
                .is_err());
        }
        assert_eq!(graph.current_state_id(), state);
        assert!(!graph.has_uncommitted_changes());
    }

    #[test]
    fn structural_id_parser_keeps_small_integers_and_rejects_invalid_rows() {
        let (mut graph, table, update) = fixture();
        let state = graph.current_state_id();
        for value in [
            AttrValue::Null,
            AttrValue::Int(-1),
            AttrValue::Float64(20.0),
            AttrValue::Text("20".into()),
        ] {
            let base = BaseTable::from_columns(HashMap::from([(
                "node_id".into(),
                BaseArray::from_attr_values(vec![value, AttrValue::Int(10)]),
            )]))
            .expect("base");
            let nodes = NodesTable::from_base_table(base).expect("nodes");
            assert!(GraphTable::new(nodes, table.edges().clone())
                .update_graph(&mut graph, state, &update)
                .is_err());
        }
        let base = BaseTable::from_columns(HashMap::from([(
            "node_id".into(),
            BaseArray::from_attr_values(vec![AttrValue::SmallInt(20), AttrValue::SmallInt(10)]),
        )]))
        .expect("base");
        let base = base
            .with_column(
                "x".into(),
                BaseArray::from_attr_values(vec![AttrValue::Int(4), AttrValue::Null]),
            )
            .expect("property");
        let nodes = NodesTable::from_base_table(base).expect("nodes");
        GraphTable::new(nodes, table.edges().clone())
            .update_graph(&mut graph, state, &update)
            .expect("compact IDs");
    }
}
