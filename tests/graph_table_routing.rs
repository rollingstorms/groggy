use groggy::{
    storage::{
        array::BaseArray,
        table::{
            BaseTable, EdgesTable, GraphTable, NodesTable, Table, ValidationPolicy,
            ValidationStrictness,
        },
    },
    AttrValue as V, Graph,
};
use std::collections::HashMap;
fn base(cols: Vec<(&str, Vec<V>)>) -> BaseTable {
    let order = cols.iter().map(|(n, _)| n.to_string()).collect();
    BaseTable::with_column_order(
        cols.into_iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v)))
            .collect(),
        order,
    )
    .unwrap()
}
fn names(cols: &[&str]) -> Vec<String> {
    cols.iter().map(|n| n.to_string()).collect()
}
fn fixture() -> GraphTable {
    GraphTable::with_policy(
        NodesTable::from_base_table(base(vec![
            ("node_id", vec![V::Int(7), V::Int(3)]),
            ("x", vec![V::Int(9), V::Int(2)]),
            ("node_group", vec![V::Int(1), V::Int(2)]),
            ("kind", vec![V::Text("a".into()); 2]),
        ]))
        .unwrap(),
        EdgesTable::from_base_table(base(vec![
            ("edge_id", vec![V::Int(9), V::Int(5)]),
            ("source", vec![V::Int(7), V::Int(3)]),
            ("target", vec![V::Int(3), V::Int(7)]),
            ("x", vec![V::Int(8), V::Int(1)]),
            ("edge_group", vec![V::Int(1), V::Int(2)]),
            ("kind", vec![V::Text("a".into()); 2]),
        ]))
        .unwrap(),
        ValidationPolicy {
            strictness: ValidationStrictness::Strict,
            ..ValidationPolicy::default()
        },
    )
}
#[test]
fn melt_resolves_identifier_and_value_columns_together_for_edges_with_shared_values() {
    let t = fixture();
    let result = t
        .melt(
            Some(&names(&["edge_id", "source", "target"])),
            Some(&names(&["x"])),
            None,
            None,
        )
        .unwrap();
    assert_eq!(result.edges().edge_ids().unwrap(), vec![9, 5]);
    assert_eq!(
        result.edges().get_column("value").unwrap().data(),
        &vec![V::Int(8), V::Int(1)]
    );
    assert_eq!(result.nodes().column_names(), t.nodes().column_names());
    assert_eq!(
        result.nodes().get_column("x").unwrap().data(),
        t.nodes().get_column("x").unwrap().data()
    );
    assert_eq!(result.policy().strictness, ValidationStrictness::Strict);
}
#[test]
fn shared_sort_keys_prefer_nodes_and_edge_only_keys_leave_nodes_unchanged() {
    let t = fixture();
    assert_eq!(
        t.sort_by("x", true).unwrap().nodes().node_ids().unwrap(),
        vec![3, 7]
    );
    assert_eq!(
        t.sort_by("x", true).unwrap().edges().edge_ids().unwrap(),
        vec![9, 5]
    );
    assert_eq!(
        t.sort_by("edge_group", false)
            .unwrap()
            .edges()
            .edge_ids()
            .unwrap(),
        vec![5, 9]
    );
    assert_eq!(
        t.sort_values(names(&["node_group", "x"]), vec![false, true])
            .unwrap()
            .nodes()
            .node_ids()
            .unwrap(),
        vec![3, 7]
    );
    assert!(t
        .sort_values(names(&["node_group", "edge_group"]), vec![true, true])
        .is_err());
}
#[test]
fn grouping_retains_domain_specific_groups_and_rejects_ambiguous_or_cross_domain_keys() {
    let t = fixture();
    let groups = t.group_by(&names(&["node_group"])).unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].nodes().node_ids().unwrap(), vec![7]);
    assert_eq!(groups[0].edges().edge_ids().unwrap(), vec![9, 5]);
    let groups = t.group_by(&names(&["edge_group"])).unwrap();
    assert_eq!(groups[0].edges().edge_ids().unwrap(), vec![9]);
    assert_eq!(groups[0].nodes().node_ids().unwrap(), vec![7, 3]);
    assert!(t
        .group_by(&names(&["x"]))
        .unwrap_err()
        .to_string()
        .contains("Ambiguous"));
    assert!(t.group_by(&names(&["node_group", "edge_group"])).is_err());
    assert!(t.group_by(&names(&["missing"])).is_err());
}
#[test]
fn pivot_resolves_all_operands_and_retains_structural_validation() {
    let t = fixture();
    let nodes = t
        .pivot_table(&names(&["node_id"]), "kind", "x", "sum")
        .unwrap();
    assert_eq!(nodes.nodes().nrows(), 2);
    assert!(nodes.nodes().has_column("kind_a"));
    assert_eq!(nodes.edges().column_names(), t.edges().column_names());
    let edges = t
        .pivot_table(&names(&["edge_id", "source", "target"]), "kind", "x", "sum")
        .unwrap();
    assert!(edges.edges().has_column("kind_a"));
    assert!(t
        .pivot_table(&names(&["node_group"]), "kind", "x", "sum")
        .is_err());
    assert!(t
        .pivot_table(&names(&["node_id"]), "edge_group", "x", "sum")
        .is_err());
}
#[test]
fn conflicting_melt_operands_reject_without_changing_either_component() {
    let t = fixture();
    assert!(t
        .melt(
            Some(&names(&["node_id"])),
            Some(&names(&["edge_group"])),
            None,
            None
        )
        .is_err());
    assert_eq!(t.nodes().node_ids().unwrap(), vec![7, 3]);
    assert_eq!(t.edges().edge_ids().unwrap(), vec![9, 5]);
    assert!(t
        .melt(
            Some(&names(&["edge_id", "source", "target"])),
            Some(&names(&["x", "edge_group"])),
            None,
            None
        )
        .is_err());
}
#[test]
fn routing_preserves_source_context_and_never_invents_row_identity_for_reshape() {
    let mut g = Graph::new_publishing();
    g.insert_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    g.set_node_attrs(HashMap::from([(
        "x".into(),
        vec![(0, V::Int(2)), (1, V::Int(1))],
    )]))
    .unwrap();
    let t = g.table().unwrap();
    let sorted = t.sort_by("x", true).unwrap();
    assert_eq!(sorted.nodes().base().source_entity_ids(), Some(&[1, 0][..]));
    let melted = t
        .melt(Some(&names(&["node_id"])), Some(&names(&["x"])), None, None)
        .unwrap();
    assert_eq!(melted.nodes().base().source_entity_ids(), None);
    assert_eq!(melted.nodes().base().source_views().len(), 1);
    assert_eq!(g.current_state_id(), 3);
}
