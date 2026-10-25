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
        NodesTable::new(vec![70, 12, 99, 44])
            .with_attributes(
                "group".into(),
                HashMap::from([
                    (70, V::Int(1)),
                    (12, V::Int(1)),
                    (99, V::Int(2)),
                    (44, V::Int(2)),
                ]),
            )
            .unwrap(),
        EdgesTable::new(vec![
            (900, 70, 12),
            (901, 70, 12),
            (902, 12, 12),
            (903, 12, 99),
            (904, 99, 44),
            (905, 44, 70),
        ]),
    )
}
#[test]
fn head_tail_and_slice_keep_internal_parallel_edges_loops_and_original_row_order() {
    let table = fixture();
    let head = table.head(2).unwrap();
    assert_eq!(head.nodes().node_ids().unwrap(), vec![70, 12]);
    assert_eq!(head.edges().edge_ids().unwrap(), vec![900, 901, 902]);
    assert_eq!(
        table.tail(2).unwrap().edges().edge_ids().unwrap(),
        vec![904]
    );
    assert_eq!(
        table.slice(1, 3).unwrap().edges().edge_ids().unwrap(),
        vec![902, 903]
    );
    assert_eq!(table.edges().nrows(), 6);
    assert!(head.to_viz_data_source(false).is_ok());
    assert!(head.to_graph().is_ok());
}
#[test]
fn empty_oversized_and_reversed_ranges_preserve_empty_schemas_and_policy() {
    let table = fixture();
    for selected in [
        table.head(0).unwrap(),
        table.tail(0).unwrap(),
        table.slice(10, usize::MAX).unwrap(),
        table.slice(3, 1).unwrap(),
    ] {
        assert_eq!(selected.shape().nodes, (0, 2));
        assert_eq!(selected.shape().edges, (0, 3));
        assert!(selected.is_empty());
        assert_eq!(selected.policy().strictness, table.policy().strictness);
    }
    let full = table.head(usize::MAX).unwrap();
    assert!(std::ptr::eq(full.edges(), table.edges()));
    assert_eq!(
        full.edges().edge_ids().unwrap(),
        table.edges().edge_ids().unwrap()
    );
}
#[test]
fn node_groups_partition_edges_once_and_exclude_cross_group_routes() {
    let table = fixture();
    let groups = table.group_by(&["group".into()]).unwrap();
    assert_eq!(groups.len(), 2);
    let mut counts = vec![];
    for group in groups {
        assert!(group.validate().is_valid());
        counts.push(group.edges().edge_ids().unwrap());
    }
    counts.sort();
    assert_eq!(counts, vec![vec![900, 901, 902], vec![904]]);
}
#[test]
fn both_components_report_their_own_shapes_and_schemas_even_with_shared_names() {
    let table = fixture();
    assert_eq!(table.shape().nodes, (4, 2));
    assert_eq!(table.shape().edges, (6, 3));
    assert_eq!(table.column_names().0, table.nodes().column_names());
    assert_eq!(table.column_names().1, table.edges().column_names());
}
#[test]
fn induced_rows_keep_the_exact_source_capture_and_publish_mixed_properties_once() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(4).unwrap();
    graph.add_edges(&[(0, 1), (1, 2), (2, 3), (0, 1)]).unwrap();
    let original = graph.table().unwrap();
    let mut head = original.head(2).unwrap();
    assert_eq!(head.edges().base().source_entity_ids(), Some(&[0, 3][..]));
    head.nodes_mut()
        .base_table_mut()
        .set_column("x", vec![V::Int(1); 2])
        .unwrap();
    head.edges_mut()
        .base_table_mut()
        .set_column("w", vec![V::Int(2); 2])
        .unwrap();
    let before = graph.current_state_id();
    head.update_source_graph(&mut graph, &["x".into()], &["w".into()], Some(123), None)
        .unwrap();
    assert_eq!(graph.current_state_id(), before + 1);
    assert!(!original.nodes().has_column("x"));
    assert_eq!(
        graph.get_edge_attr(3, &"w".into()).unwrap(),
        Some(V::Int(2))
    );
}
#[test]
fn malformed_or_dangling_input_rejects_instead_of_disappearing_in_an_empty_subset() {
    let dangling = GraphTable::new(NodesTable::new(vec![1]), EdgesTable::new(vec![(1, 1, 2)]));
    assert!(dangling.head(0).is_err());
    let duplicate = GraphTable::new(NodesTable::new(vec![1, 1]), EdgesTable::new(vec![]));
    assert!(duplicate.tail(1).is_err());
    let bad_edges = EdgesTable::from_base_table(
        BaseTable::from_columns(HashMap::from([
            (
                "edge_id".into(),
                BaseArray::from_attr_values(vec![V::Int(1)]),
            ),
            (
                "source".into(),
                BaseArray::from_attr_values(vec![V::Int(-1)]),
            ),
            (
                "target".into(),
                BaseArray::from_attr_values(vec![V::Int(1)]),
            ),
        ]))
        .unwrap(),
    )
    .unwrap();
    assert!(GraphTable::new(NodesTable::new(vec![1]), bad_edges)
        .head(0)
        .is_err());
}

#[test]
fn middle_and_reversed_base_ranges_keep_column_lengths_and_source_rows_aligned() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(4).unwrap();
    let original = graph.nodes_table().unwrap();
    let middle = original.base().slice(1, 3);
    assert_eq!(middle.nrows(), 2);
    assert_eq!(middle.source_entity_ids(), Some(&[1, 2][..]));
    assert_eq!(middle.column("node_id").unwrap().len(), 2);
    assert_eq!(
        middle
            .clone()
            .to_nodes("node_id")
            .unwrap()
            .node_ids()
            .unwrap(),
        vec![1, 2]
    );
    let reversed = original.base().slice(3, 1);
    assert_eq!(reversed.nrows(), 0);
    assert_eq!(reversed.source_entity_ids(), Some(&[][..]));
    assert!(reversed.column("node_id").unwrap().is_empty());
}
