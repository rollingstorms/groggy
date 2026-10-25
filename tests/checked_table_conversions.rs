use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue as V, Graph,
};
use std::{collections::HashMap, sync::Arc};

fn base(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    let order = columns.iter().map(|(name, _)| name.to_string()).collect();
    BaseTable::with_column_order(
        columns
            .into_iter()
            .map(|(name, values)| (name.to_string(), BaseArray::from_attr_values(values)))
            .collect(),
        order,
    )
    .unwrap()
}
fn names(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

#[test]
fn custom_node_key_renames_without_remapping_values_or_copying_column_buffers() {
    let input = base(vec![
        ("uid", vec![V::SmallInt(70), V::Int(12)]),
        ("value", vec![V::Float64(1.23456789012345), V::Null]),
    ]);
    let nodes = input.clone().to_nodes("uid").unwrap();
    assert_eq!(nodes.node_ids().unwrap(), vec![70, 12]);
    assert_eq!(nodes.column_names(), &names(&["node_id", "value"]));
    assert!(Arc::ptr_eq(
        &input.column("uid").unwrap().inner,
        &nodes.column("node_id").unwrap().inner
    ));
    assert!(Arc::ptr_eq(
        &input.column("value").unwrap().inner,
        &nodes.column("value").unwrap().inner
    ));
    assert_eq!(input.column_names(), &names(&["uid", "value"]));
}

#[test]
fn explicit_conversion_rejects_unknown_keys_collisions_and_invalid_node_identity() {
    let collision = base(vec![("node_id", vec![V::Int(1)]), ("uid", vec![V::Int(2)])]);
    assert!(collision
        .clone()
        .to_nodes("uid")
        .unwrap_err()
        .to_string()
        .contains("Duplicate column"));
    assert!(collision.to_nodes("missing").is_err());
    for values in [
        vec![V::Null],
        vec![V::Int(-1)],
        vec![V::Float64(1.0)],
        vec![V::Text("1".into())],
        vec![V::Bool(true)],
        vec![V::Int(7), V::SmallInt(7)],
    ] {
        let input = base(vec![("uid", values)]);
        assert!(input.clone().to_nodes("uid").is_err());
        assert!(input.has_column("uid"));
    }
}

#[test]
fn checked_edges_normalize_existing_aliases_and_keep_parallel_edges_and_loops() {
    let input = base(vec![
        ("edge_ids", vec![V::Int(91), V::SmallInt(92), V::Int(93)]),
        ("src", vec![V::Int(70), V::Int(70), V::Int(12)]),
        ("dst", vec![V::Int(12); 3]),
        ("weight", vec![V::Float64(3.75), V::Null, V::SmallInt(2)]),
    ]);
    let edges = input.clone().to_edges().unwrap();
    assert_eq!(
        edges.as_tuples().unwrap(),
        vec![(91, 70, 12), (92, 70, 12), (93, 12, 12)]
    );
    assert_eq!(
        edges.column("weight").unwrap().data(),
        input.column("weight").unwrap().data()
    );
    assert!(input.has_column("src"));
}

#[test]
fn malformed_endpoints_and_duplicate_edges_reject_without_allocating_identity() {
    for (name, value) in [
        ("edge_id", V::Null),
        ("source", V::Int(-1)),
        ("target", V::Float64(1.0)),
    ] {
        let input = base(
            ["edge_id", "source", "target"]
                .into_iter()
                .map(|n| (n, vec![if n == name { value.clone() } else { V::Int(1) }]))
                .collect(),
        );
        assert!(input.to_edges().is_err());
    }
    let input = base(vec![
        ("edge_id", vec![V::Int(7); 2]),
        ("source", vec![V::Int(70); 2]),
        ("target", vec![V::Int(12); 2]),
    ]);
    assert!(input
        .to_edges()
        .unwrap_err()
        .to_string()
        .contains("Duplicate edge_id"));
}

#[test]
fn exact_captured_identity_survives_renaming_and_publishes_one_checked_update() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(2).unwrap();
    graph.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    let original = graph.nodes_table().unwrap();
    let renamed = original
        .base()
        .rename(HashMap::from([("node_id".into(), "uid".into())]))
        .unwrap();
    let mut nodes = renamed.to_nodes("uid").unwrap();
    assert_eq!(nodes.base().source_entity_kind(), Some("node"));
    assert_eq!(nodes.base().source_entity_ids(), Some(&[0, 1][..]));
    assert_eq!(
        nodes.base().source_state_id(),
        original.base().source_state_id()
    );
    nodes
        .base_table_mut()
        .set_column("x", vec![V::Int(8), V::Int(9)])
        .unwrap();
    let before = graph.current_state_id();
    nodes
        .base()
        .update_source_graph(&mut graph, &names(&["x"]), None, None)
        .unwrap();
    assert_eq!(graph.current_state_id(), before + 1);
    assert_eq!(
        graph.get_node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(8))
    );
    assert_eq!(original.column("x").unwrap().data()[0], V::Int(1));
}

#[test]
fn changed_identity_kind_or_endpoints_retains_only_read_only_source_context() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    let original = graph.edges_table().unwrap();
    let nodes = original
        .base()
        .select(&names(&["source"]))
        .unwrap()
        .to_nodes("source")
        .unwrap();
    assert_eq!(nodes.base().source_entity_kind(), None);
    assert_eq!(nodes.base().source_views().len(), 1);
    let changed = original
        .base()
        .with_column(
            "target".into(),
            BaseArray::from_attr_values(vec![V::Int(0)]),
        )
        .unwrap();
    let edges = changed.to_edges().unwrap();
    assert_eq!(edges.base().source_entity_ids(), None);
    assert_eq!(
        edges.base().source_views()[0].node_ids(),
        original.base().source_views()[0].node_ids()
    );
    assert!(edges
        .base()
        .update_source_graph(&mut graph, &names(&[]), None, None)
        .is_err());
    assert_eq!(graph.edge_endpoints(0).unwrap(), (0, 1));
    let altered_ids = graph
        .nodes_table()
        .unwrap()
        .base()
        .with_column(
            "uid".into(),
            BaseArray::from_attr_values(vec![V::Int(70), V::Int(12)]),
        )
        .unwrap()
        .select(&names(&["uid"]))
        .unwrap()
        .to_nodes("uid")
        .unwrap();
    assert_eq!(altered_ids.base().source_entity_ids(), None);
    assert_eq!(altered_ids.base().source_views().len(), 1);
}

#[test]
fn empty_structural_tables_convert_without_inventing_rows() {
    assert_eq!(
        base(vec![("uid", vec![])]).to_nodes("uid").unwrap().nrows(),
        0
    );
    assert_eq!(
        base(vec![
            ("edge_id", vec![]),
            ("source", vec![]),
            ("target", vec![])
        ])
        .to_edges()
        .unwrap()
        .nrows(),
        0
    );
}
