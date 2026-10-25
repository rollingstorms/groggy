use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, NodesTable, Table},
    },
    AttrValue as V, Graph,
};
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
fn nodes() -> NodesTable {
    NodesTable::from_base_table(base(vec![("node_id", vec![V::SmallInt(7), V::Int(3)])])).unwrap()
}
fn edges() -> EdgesTable {
    EdgesTable::from_base_table(base(vec![
        ("edge_id", vec![V::SmallInt(9), V::Int(5)]),
        ("source", vec![V::SmallInt(7), V::Int(3)]),
        ("target", vec![V::Int(3), V::SmallInt(7)]),
    ]))
    .unwrap()
}
#[test]
fn mapped_columns_preserve_physical_order_null_unspecified_rows_and_typed_values() {
    let n = nodes()
        .with_attribute_entries("x".into(), vec![(3, V::Float64(1.0000000000000002))])
        .unwrap();
    assert_eq!(
        n.column("x").unwrap().data(),
        &vec![V::Null, V::Float64(1.0000000000000002)]
    );
    let e = edges()
        .with_attribute_entries("v".into(), vec![(5, V::IntVec(vec![2, 3]))])
        .unwrap();
    assert_eq!(
        e.column("v").unwrap().data(),
        &vec![V::Null, V::IntVec(vec![2, 3])]
    );
    assert!(!nodes().has_column("x"));
    assert!(!edges().has_column("v"));
    assert_eq!(
        n.with_attributes("x".into(), Default::default())
            .unwrap()
            .column("x")
            .unwrap()
            .data(),
        &vec![V::Null, V::Null]
    );
}
#[test]
fn attribute_mapping_rejects_unknown_duplicate_and_ambiguous_source_ids() {
    for entries in [vec![(999, V::Int(1))], vec![(7, V::Int(1)), (7, V::Int(2))]] {
        assert!(nodes().with_attribute_entries("x".into(), entries).is_err());
    }
    assert!(edges()
        .with_attribute_entries("x".into(), vec![(999, V::Int(1))])
        .is_err());
    assert!(edges()
        .with_attribute_entries("x".into(), vec![(9, V::Int(1)), (9, V::Int(2))])
        .is_err());
    let duplicate =
        NodesTable::from_base_table(base(vec![("node_id", vec![V::Int(1), V::Int(1)])])).unwrap();
    assert!(duplicate
        .with_attribute_entries("x".into(), vec![])
        .is_err());
    assert!(duplicate.validate_uids().is_err());
}
#[test]
fn structural_reads_are_aligned_and_accept_small_integers_without_wrapping() {
    let n = nodes();
    assert_eq!(n.node_ids().unwrap(), vec![7, 3]);
    n.validate_uids().unwrap();
    assert!(n.get_by_uid(7).is_some());
    assert_eq!(
        n.iter_with_ids().map(|(id, _)| id).collect::<Vec<_>>(),
        vec![7, 3]
    );
    let e = edges();
    assert_eq!(e.as_tuples().unwrap(), vec![(9, 7, 3), (5, 3, 7)]);
    assert!(e.get_by_edge_id(9).is_some());
    assert_eq!(
        e.iter_with_edge_info()
            .map(|(id, s, t, _)| (id, s, t))
            .collect::<Vec<_>>(),
        e.as_tuples().unwrap()
    );
    assert_eq!(
        e.filter_by_sources(&[3]).unwrap().edge_ids().unwrap(),
        vec![5]
    );
    assert_eq!(
        e.filter_by_targets(&[3]).unwrap().edge_ids().unwrap(),
        vec![9]
    );
    assert_eq!(e.filter_by_sources(&[]).unwrap().nrows(), 0);
}
#[test]
fn malformed_structural_columns_reject_instead_of_compacting_row_alignment() {
    for value in [
        V::Null,
        V::Int(-1),
        V::Float64(7.0),
        V::Text("7".into()),
        V::Bool(true),
    ] {
        let n =
            NodesTable::from_base_table(base(vec![("node_id", vec![value.clone(), V::Int(3)])]))
                .unwrap();
        assert!(n.node_ids().is_err());
        assert!(n.validate_uids().is_err());
        assert!(n.get_by_uid(usize::MAX).is_none());
        for name in ["edge_id", "source", "target"] {
            let mut e = edges();
            e.base_table_mut()
                .set_column(name, vec![value.clone(), V::Int(3)])
                .unwrap();
            assert!(e.as_tuples().is_err());
            if name == "source" {
                assert!(e.sources().is_err());
                assert!(e.filter_by_sources(&[3]).is_err());
            }
            if name == "target" {
                assert!(e.targets().is_err());
                assert!(e.filter_by_targets(&[3]).is_err());
            }
        }
    }
}
#[test]
fn table_property_edits_retain_captured_row_identity_without_mutating_graph() {
    let mut graph = Graph::new();
    graph.add_nodes(2);
    let edge = graph.add_edge(0, 1).unwrap();
    let table = graph.read_view().unwrap().table().unwrap();
    let n = table
        .nodes()
        .with_attribute_entries("x".into(), vec![(1, V::Int(3))])
        .unwrap();
    let e = table
        .edges()
        .with_attribute_entries("w".into(), vec![(edge, V::Int(4))])
        .unwrap();
    assert_eq!(n.base().source_entity_ids(), Some(&[0, 1][..]));
    assert_eq!(e.base().source_entity_ids(), Some(&[edge][..]));
    assert_eq!(graph.get_node_attr(1, &"x".into()).unwrap(), None);
    assert_eq!(graph.get_edge_attr(edge, &"w".into()).unwrap(), None);
}
#[test]
fn wide_integer_ids_and_empty_attribute_inputs_remain_exact() {
    let wide = (1usize << 53) + 1;
    let e = EdgesTable::new(vec![(wide, wide + 2, wide + 4)]);
    assert_eq!(e.as_tuples().unwrap(), vec![(wide, wide + 2, wide + 4)]);
    assert_eq!(
        e.with_attribute_entries("x".into(), vec![(wide, V::Int(1))])
            .unwrap()
            .edge_ids()
            .unwrap(),
        vec![wide]
    );
    let n = NodesTable::new(vec![])
        .with_attribute_entries("x".into(), vec![])
        .unwrap();
    assert!(n.has_column("x"));
    assert_eq!(n.nrows(), 0);
    assert!(n
        .with_attribute_entries("x".into(), vec![(0, V::Int(1))])
        .is_err());
}
