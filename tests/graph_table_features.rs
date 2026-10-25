use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable},
    },
    AttrValue as V,
};

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
fn fixture(rows: usize, features: usize) -> GraphTable {
    let mut nodes = vec![(
        "node_id",
        (0..rows).map(|r| V::Int((r * 7 + 3) as i64)).collect(),
    )];
    let mut edges = vec![
        (
            "edge_id",
            (0..rows).map(|r| V::Int((r * 11 + 2) as i64)).collect(),
        ),
        ("source", vec![V::Int(3); rows]),
        ("target", vec![V::Int(3); rows]),
    ];
    for name in ["z", "a", "m"].into_iter().take(features) {
        let values = (0..rows)
            .map(|r| V::Int((r * 100 + nodes.len()) as i64))
            .collect::<Vec<_>>();
        nodes.push((name, values.clone()));
        edges.push((name, values));
    }
    GraphTable::new(
        NodesTable::from_base_table(base(nodes)).unwrap(),
        EdgesTable::from_base_table(base(edges)).unwrap(),
    )
}
#[test]
fn all_small_shapes_preserve_row_major_values_names_and_node_labels() {
    for rows in 0..8 {
        for cols in 0..4 {
            let table = fixture(rows, cols);
            let nodes = table.to_matrix_f64().unwrap();
            let edges = table.edges_to_matrix::<f64>().unwrap();
            assert_eq!(nodes.shape(), (rows, cols));
            assert_eq!(edges.shape(), (rows, cols));
            let names = ["z", "a", "m"][..cols]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            assert_eq!(nodes.column_names(), names);
            assert_eq!(edges.column_names(), names);
            let labels = (0..rows)
                .map(|r| format!("node_{}", r * 7 + 3))
                .collect::<Vec<_>>();
            assert_eq!(nodes.row_labels(), Some(labels.as_slice()));
            assert_eq!(edges.row_labels(), None);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(nodes.get(r, c), Some((r * 100 + c + 1) as f64));
                    assert_eq!(edges.get(r, c), nodes.get(r, c));
                }
            }
        }
    }
}
#[test]
fn conversions_share_dtype_null_text_and_boolean_policy() {
    let table = GraphTable::new(
        NodesTable::from_base_table(base(vec![
            ("node_id", vec![V::Int(10), V::Int(20), V::Int(30)]),
            ("x", vec![V::Null, V::Bool(true), V::Text("2.5".into())]),
            (
                "y",
                vec![V::Float64(1.0000000000000002), V::Int(4), V::Int(8)],
            ),
        ]))
        .unwrap(),
        EdgesTable::new(vec![]),
    );
    let matrix = table.to_matrix_f64().unwrap();
    assert_eq!(matrix.get(0, 0), Some(0.0));
    assert_eq!(matrix.get(1, 0), Some(1.0));
    assert_eq!(matrix.get(2, 0), Some(2.5));
    assert_eq!(matrix.get(0, 1), Some(1.0000000000000002));
    assert!(table.to_matrix_i64().is_err()); // decimal text is not an integer
    assert_eq!(table.to_matrix_f32().unwrap().get(2, 0), Some(2.5));
}
#[test]
fn invalid_feature_reports_column_and_row_without_mutating_input() {
    let mut table = fixture(2, 1);
    table
        .nodes_mut()
        .base_table_mut()
        .set_column("z", vec![V::Int(1), V::Text("bad".into())])
        .unwrap();
    let error = table.to_matrix_f64().unwrap_err().to_string();
    assert!(error.contains("z row 1"), "{error}");
    assert!(table.to_matrix_f64().is_err());
}
#[test]
fn invalid_or_duplicate_ids_reject_even_without_features() {
    for ids in [
        vec![V::Int(-1)],
        vec![V::Float64(1.0)],
        vec![V::Null],
        vec![V::Int(1), V::Int(1)],
    ] {
        let table = GraphTable::new(
            NodesTable::from_base_table(base(vec![("node_id", ids.clone())])).unwrap(),
            EdgesTable::from_base_table(base(vec![
                ("edge_id", ids.clone()),
                ("source", vec![V::Int(0); ids.len()]),
                ("target", vec![V::Int(0); ids.len()]),
            ]))
            .unwrap(),
        );
        assert!(table.to_matrix_f64().is_err());
        assert!(table.edges_to_matrix::<f64>().is_err());
    }
}

#[test]
fn matrices_are_owned_copies_and_range_failures_are_checked() {
    let mut table = fixture(2, 1);
    let matrix = table.to_matrix_f64().unwrap();
    table
        .nodes_mut()
        .base_table_mut()
        .set_column("z", vec![V::Int(99), V::Int(99)])
        .unwrap();
    assert_eq!(matrix.get(0, 0), Some(1.0));
    assert_eq!(table.to_matrix_f64().unwrap().get(0, 0), Some(99.0));
    table
        .nodes_mut()
        .base_table_mut()
        .set_column("z", vec![V::Float64(f64::MAX); 2])
        .unwrap();
    assert!(table.to_matrix_f32().is_err());
    assert!(table.to_matrix_i64().is_err());
    table
        .edges_mut()
        .base_table_mut()
        .set_column("z", vec![V::Text("bad".into()); 2])
        .unwrap();
    let error = table.edges_to_matrix::<f64>().unwrap_err().to_string();
    assert!(error.contains("z row 0"), "{error}");
}
