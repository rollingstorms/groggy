use groggy::{
    storage::table::{GraphTable, Table},
    subgraphs::Subgraph,
    AttrValue as V, Graph,
};
use std::collections::HashSet;
fn fixture() -> Graph {
    let mut g = Graph::new_publishing();
    g.insert_nodes(3).unwrap();
    g.add_edges(&[(0, 1), (0, 1), (1, 2)]).unwrap();
    g.set_node_attr(0, "x".into(), V::Float64(1.0000000000000002))
        .unwrap();
    g.set_node_attr(1, "explicit_null".into(), V::Null).unwrap();
    g.set_edge_attr(1, "weight".into(), V::Int(7)).unwrap();
    g
}
fn verify(table: &GraphTable) {
    assert_eq!(table.nodes().base().source_entity_kind(), Some("node"));
    assert_eq!(table.edges().base().source_entity_kind(), Some("edge"));
    assert_eq!(table.nodes().base().source_views().len(), 1);
    assert_eq!(table.edges().base().source_views().len(), 1);
}
#[test]
fn direct_graph_exports_retain_frozen_source_values_and_publish_mixed_columns_once() {
    let mut g = fixture();
    let mut table = g.table().unwrap();
    verify(&table);
    assert_eq!(
        table.nodes().base().source_entity_ids(),
        Some(&[0, 1, 2][..])
    );
    table
        .nodes_mut()
        .base_table_mut()
        .set_column("score", vec![V::Int(10); 3])
        .unwrap();
    table
        .edges_mut()
        .base_table_mut()
        .set_column("weight", vec![V::Int(8); 3])
        .unwrap();
    let before = g.current_state_id();
    let result = table
        .update_source_graph(
            &mut g,
            &["score".into()],
            &["weight".into()],
            Some(123),
            None,
        )
        .unwrap();
    assert_eq!(result.batch.state_id, before + 1);
    assert_eq!(
        table.nodes().base().source_views()[0]
            .node_attr(0, &"score".into())
            .unwrap(),
        None
    );
    assert_eq!(
        g.get_edge_attr(1, &"weight".into()).unwrap(),
        Some(V::Int(8))
    );
}
#[test]
fn separate_graph_exports_keep_independent_capture_identity_and_exact_properties() {
    let mut g = fixture();
    let n = g.nodes_table().unwrap();
    let e = g.edges_table().unwrap();
    assert_eq!(
        n.column("x").unwrap().get(0),
        Some(&V::Float64(1.0000000000000002))
    );
    assert!(n.has_column("explicit_null"));
    let t = GraphTable::new(n, e);
    assert!(t
        .update_source_graph(&mut g, &["x".into()], &["weight".into()], None, None)
        .is_err());
}
#[test]
fn connected_subgraph_exports_pin_exact_selection_and_remain_readable_after_parent_mutation() {
    let graph = fixture().into_shared();
    let sg = Subgraph::new(
        graph.clone(),
        HashSet::from([0, 1]),
        HashSet::from([1]),
        "selected".into(),
    );
    let table = sg.table().unwrap();
    verify(&table);
    assert_eq!(table.edges().edge_ids().unwrap(), vec![1]);
    assert_eq!(table.nodes().node_ids().unwrap(), vec![0, 1]);
    graph
        .borrow_mut()
        .set_node_attr(0, "x".into(), V::Int(99))
        .unwrap();
    assert_eq!(
        table.nodes().base().source_views()[0]
            .node_attr(0, &"x".into())
            .unwrap(),
        Some(V::Float64(1.0000000000000002))
    );
    assert_eq!(
        sg.nodes_table().unwrap().column("x").unwrap().get(0),
        Some(&V::Int(99))
    );
}
#[test]
fn stale_subgraph_exports_reject_instead_of_synthesizing_rows() {
    let graph = fixture().into_shared();
    let sg = Subgraph::new(
        graph.clone(),
        HashSet::from([0, 1]),
        HashSet::from([1]),
        "selected".into(),
    );
    let old = sg.table().unwrap();
    graph.borrow_mut().remove_edge(1).unwrap();
    assert!(sg.table().is_err());
    assert!(sg.nodes_table().is_err());
    assert!(sg.edges_table().is_err());
    assert_eq!(
        old.edges().base().source_views()[0]
            .edge_endpoints(1)
            .unwrap(),
        (0, 1)
    );
}
#[test]
fn exports_capture_working_manual_values_but_do_not_authorize_dirty_publication() {
    let mut g = Graph::new();
    g.add_node();
    g.set_node_attr(0, "x".into(), V::Int(3)).unwrap();
    let table = g.nodes_table().unwrap();
    assert_eq!(table.base().source_state_id(), None);
    assert_eq!(table.column("x").unwrap().get(0), Some(&V::Int(3)));
    assert!(table
        .base()
        .update_source_graph(&mut g, &["x".into()], None, None)
        .is_err());
}
#[test]
fn empty_graph_and_subgraph_exports_keep_source_kinds_and_empty_row_identity() {
    let g = Graph::new();
    let t = g.table().unwrap();
    verify(&t);
    assert_eq!(t.nodes().base().source_entity_ids(), Some(&[][..]));
    let graph = fixture().into_shared();
    let sg = Subgraph::new(graph, HashSet::new(), HashSet::new(), "empty".into());
    let t = sg.table().unwrap();
    verify(&t);
    assert!(t.is_empty());
    assert_eq!(t.edges().base().source_entity_ids(), Some(&[][..]));
}
