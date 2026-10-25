use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable, Table},
    },
    AttrValue as V, Graph,
};
use std::collections::HashMap;
fn columns(rows: usize) -> HashMap<String, BaseArray<V>> {
    HashMap::from([
        (
            "x".into(),
            BaseArray::from_attr_values(vec![V::Int(1); rows]),
        ),
        (
            "y".into(),
            BaseArray::from_attr_values(vec![V::Int(2); rows]),
        ),
    ])
}
#[test]
fn ordered_constructor_rejects_hidden_duplicate_and_unknown_columns_even_when_empty() {
    for rows in [0, 2] {
        for order in [vec!["x"], vec!["x", "x"], vec!["x", "ghost"], vec![]] {
            assert!(BaseTable::with_column_order(
                columns(rows),
                order.iter().map(|s| s.to_string()).collect()
            )
            .is_err());
        }
        let t = BaseTable::with_column_order(columns(rows), vec!["y".into(), "x".into()]).unwrap();
        assert_eq!(t.column_names(), &["y".to_string(), "x".to_string()]);
        assert_eq!(t.nrows(), rows);
    }
}
#[test]
fn truly_empty_table_can_initialize_but_empty_declared_schema_cannot_expand_one_column() {
    let empty = BaseTable::new();
    let grown = empty
        .with_column("x".into(), BaseArray::from_attr_values(vec![V::Int(1); 3]))
        .unwrap();
    assert_eq!(grown.nrows(), 3);
    let declared = BaseTable::with_column_order(columns(0), vec!["x".into(), "y".into()]).unwrap();
    assert!(declared
        .with_column("x".into(), BaseArray::from_attr_values(vec![V::Int(1)]))
        .is_err());
    assert!(declared
        .with_column("z".into(), BaseArray::from_attr_values(vec![V::Int(1)]))
        .is_err());
    let valid = declared
        .with_column("z".into(), BaseArray::from_attr_values(vec![]))
        .unwrap();
    assert_eq!(valid.nrows(), 0);
    assert_eq!(valid.ncols(), 3);
}
#[test]
fn zero_column_rows_remain_rectangular_when_columns_are_added() {
    let table = BaseTable::with_column_order(columns(2), vec!["x".into(), "y".into()])
        .unwrap()
        .select(&[])
        .unwrap();
    assert_eq!(table.nrows(), 2);
    assert!(table
        .with_column("z".into(), BaseArray::from_attr_values(vec![V::Int(1); 3]))
        .is_err());
    let valid = table
        .with_column("z".into(), BaseArray::from_attr_values(vec![V::Int(1); 2]))
        .unwrap();
    assert_eq!(valid.nrows(), 2);
    assert_eq!(valid.ncols(), 1);
}
#[test]
fn all_projection_layers_reject_duplicate_or_unknown_names_before_output() {
    let n = NodesTable::new(vec![0, 1]);
    let e = EdgesTable::new(vec![(0, 0, 1)]);
    let g = GraphTable::new(n.clone(), e.clone());
    for names in [
        vec!["node_id".into(), "node_id".into()],
        vec!["ghost".into()],
    ] {
        assert!(n.select(&names).is_err());
        assert!(g.select(&names).is_err());
        assert!(n.base().select(&names).is_err());
    }
    assert!(e.select(&["source".into(), "source".into()]).is_err());
    let empty = g.select(&[]).unwrap();
    assert_eq!(empty.nodes().nrows(), 2);
    assert_eq!(empty.edges().nrows(), 1);
}
#[test]
fn rejected_projection_and_column_edits_leave_source_capture_unchanged() {
    let mut g = Graph::new_publishing();
    g.insert_nodes(2).unwrap();
    let table = g.nodes_table().unwrap().into_base();
    assert!(table.select(&["node_id".into(), "node_id".into()]).is_err());
    assert!(table
        .with_column("z".into(), BaseArray::from_attr_values(vec![]))
        .is_err());
    assert_eq!(table.source_entity_ids(), Some(&[0, 1][..]));
    assert_eq!(g.current_state_id(), 1);
}
#[test]
fn sparse_from_columns_padding_keeps_its_explicit_constructor_behavior() {
    let table = BaseTable::from_columns(HashMap::from([
        (
            "x".into(),
            BaseArray::from_attr_values(vec![V::Int(1), V::Int(2)]),
        ),
        ("y".into(), BaseArray::from_attr_values(vec![V::Int(3)])),
    ]))
    .unwrap();
    assert_eq!(table.nrows(), 2);
    assert_eq!(table.column("y").unwrap().data(), &vec![V::Int(3), V::Null]);
}
