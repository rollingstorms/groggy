use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable, Table},
    },
    AttrValue as V, Graph,
};
use std::collections::HashMap;
fn fixture() -> GraphTable {
    GraphTable::new(
        NodesTable::new(vec![0, 1])
            .with_attributes("g".into(), HashMap::from([(0, V::Int(0)), (1, V::Int(1))]))
            .unwrap(),
        EdgesTable::new(vec![(0, 0, 1)])
            .with_attributes("w".into(), HashMap::from([(0, V::Int(3))]))
            .unwrap(),
    )
}
#[test]
fn clones_share_immutable_components_and_detach_only_the_mutably_accessed_component() {
    let original = fixture();
    let mut copy = original.clone();
    assert!(std::ptr::eq(original.nodes(), copy.nodes()));
    assert!(std::ptr::eq(original.edges(), copy.edges()));
    copy.nodes_mut();
    assert!(std::sync::Arc::ptr_eq(
        &copy.nodes().get_column("g").unwrap().inner,
        &original.nodes().get_column("g").unwrap().inner,
    ));
    copy.nodes_mut()
        .base_table_mut()
        .set_column("g", vec![V::Int(9); 2])
        .unwrap();
    assert!(!std::ptr::eq(original.nodes(), copy.nodes()));
    assert!(std::ptr::eq(original.edges(), copy.edges()));
    assert_eq!(
        original.nodes().get_column("g").unwrap().get(0),
        Some(&V::Int(0))
    );
    copy.edges_mut()
        .base_table_mut()
        .set_column("w", vec![V::Int(8)])
        .unwrap();
    assert_eq!(
        original.edges().get_column("w").unwrap().get(0),
        Some(&V::Int(3))
    );
}
#[test]
fn grouped_and_selected_results_share_untouched_components_without_cross_mutation() {
    let original = fixture();
    let mut groups = original.group_by(&["g".into()]).unwrap();
    assert!(groups
        .iter()
        .all(|group| std::ptr::eq(group.edges(), original.edges())));
    groups[0]
        .edges_mut()
        .base_table_mut()
        .set_column("w", vec![V::Int(99)])
        .unwrap();
    assert_eq!(
        groups[1].edges().get_column("w").unwrap().get(0),
        Some(&V::Int(3))
    );
    assert_eq!(
        original.edges().get_column("w").unwrap().get(0),
        Some(&V::Int(3))
    );
    let head = original.head(1);
    assert!(std::ptr::eq(head.edges(), original.edges()));
    assert_eq!(head.edges().nrows(), 1);
    let sorted = original.sort_by("g", false).unwrap();
    assert!(std::ptr::eq(sorted.edges(), original.edges()));
    assert_eq!(sorted.nodes().node_ids().unwrap(), vec![1, 0]);
}
#[test]
fn consuming_shared_components_returns_independent_owned_tables() {
    let original = fixture();
    let (mut nodes, mut edges) = original.clone().into_components();
    nodes.set_column("g", vec![V::Int(10); 2]).unwrap();
    edges.set_column("w", vec![V::Int(20)]).unwrap();
    assert_eq!(
        original.nodes().get_column("g").unwrap().get(0),
        Some(&V::Int(0))
    );
    assert_eq!(
        original.edges().get_column("w").unwrap().get(0),
        Some(&V::Int(3))
    );
    let unique = fixture();
    let (nodes, edges) = unique.into_components();
    assert_eq!(nodes.nrows(), 2);
    assert_eq!(edges.nrows(), 1);
}
#[test]
fn explicit_id_assignment_does_not_modify_shared_original_edge_records() {
    let base = BaseTable::with_column_order(
        HashMap::from([
            ("edge_id".into(), BaseArray::from_attr_values(vec![V::Null])),
            (
                "source".into(),
                BaseArray::from_attr_values(vec![V::Int(0)]),
            ),
            (
                "target".into(),
                BaseArray::from_attr_values(vec![V::Int(0)]),
            ),
        ]),
        vec!["edge_id".into(), "source".into(), "target".into()],
    )
    .unwrap();
    let original = GraphTable::new(
        NodesTable::new(vec![0]),
        EdgesTable::from_base_table(base).unwrap(),
    );
    let assigned = original.clone().auto_assign_edge_ids().unwrap();
    assert!(original.edges().edge_ids().is_err());
    assert_eq!(assigned.edges().edge_ids().unwrap(), vec![1]);
}
#[test]
fn detached_component_edits_preserve_original_source_binding_and_publication_scope() {
    let mut g = Graph::new_publishing();
    g.insert_nodes(2).unwrap();
    g.add_edge(0, 1).unwrap();
    let original = g.table().unwrap();
    let mut edited = original.clone();
    edited
        .nodes_mut()
        .base_table_mut()
        .set_column("x", vec![V::Int(10), V::Int(20)])
        .unwrap();
    assert_eq!(edited.nodes().base().source_entity_ids(), Some(&[0, 1][..]));
    let before = g.current_state_id();
    edited
        .update_source_graph(&mut g, &["x".into()], &[], None, None)
        .unwrap();
    assert_eq!(g.current_state_id(), before + 1);
    assert!(!original.nodes().has_column("x"));
    assert_eq!(
        original.nodes().base().source_views()[0]
            .node_attr(0, &"x".into())
            .unwrap(),
        None
    );
}
