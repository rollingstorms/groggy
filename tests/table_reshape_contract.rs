use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, NodesTable, Table},
    },
    AttrValue as V, Graph,
};
fn table(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    let order = columns.iter().map(|(n, _)| n.to_string()).collect();
    BaseTable::with_column_order(
        columns
            .into_iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v)))
            .collect(),
        order,
    )
    .unwrap()
}
fn strings(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| n.to_string()).collect()
}
#[test]
fn typed_filters_preserve_vectors_nulls_exact_numbers_and_literal_column_names() {
    let values = vec![
        V::Text("a\" == null or x == \"b".into()),
        V::Null,
        V::IntVec(vec![1, 2]),
        V::Float64(1.0000000000000002),
        V::SmallInt(7),
    ];
    let base = table(vec![
        (
            "node_id",
            (0..values.len()).map(|i| V::Int(i as i64)).collect(),
        ),
        ("odd column", values.clone()),
    ]);
    let nodes = NodesTable::from_base_table(base.clone()).unwrap();
    let edges = EdgesTable::from_base_table(
        base.rename(
            [("node_id", "edge_id")]
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        )
        .unwrap()
        .with_column(
            "source".into(),
            BaseArray::from_attr_values(vec![V::Int(0); values.len()]),
        )
        .unwrap()
        .with_column(
            "target".into(),
            BaseArray::from_attr_values(vec![V::Int(0); values.len()]),
        )
        .unwrap(),
    )
    .unwrap();
    for (row, value) in values.iter().enumerate() {
        assert_eq!(
            nodes
                .filter_by_attr("odd column", value)
                .unwrap()
                .node_ids()
                .unwrap(),
            vec![row]
        );
        assert_eq!(
            edges
                .filter_by_attr("odd column", value)
                .unwrap()
                .edge_ids()
                .unwrap(),
            vec![row]
        );
    }
    assert!(nodes.filter_by_attr("absent", &V::Null).is_err());
}
#[test]
fn pivot_groups_have_stable_first_occurrence_order_and_exact_aggregates() {
    let base = table(vec![
        ("g", vec![V::Int(20), V::Int(10), V::Int(20)]),
        ("k", vec![V::Text("x".into()); 3]),
        ("v", vec![V::Int(2), V::Int(4), V::Int(8)]),
    ]);
    for _ in 0..8 {
        let pivot = base.pivot_table(&strings(&["g"]), "k", "v", "sum").unwrap();
        assert_eq!(
            pivot.column("g").unwrap().data(),
            &vec![V::Int(20), V::Int(10)]
        );
        assert_eq!(
            pivot.column("k_x").unwrap().data(),
            &vec![V::Float64(10.0), V::Float64(4.0)]
        );
    }
    assert!(base
        .pivot_table(&strings(&["g", "g"]), "k", "v", "sum")
        .is_err());
}
#[test]
fn melt_rejects_duplicate_overlap_and_output_name_collisions_even_when_empty() {
    for rows in [0, 1] {
        let base = table(vec![
            ("id", vec![V::Int(1); rows]),
            ("x", vec![V::Int(2); rows]),
        ]);
        for (ids, values, var, value) in [
            (strings(&["id", "id"]), strings(&["x"]), "variable", "value"),
            (strings(&["id"]), strings(&["x", "x"]), "variable", "value"),
            (strings(&["id"]), strings(&["id"]), "variable", "value"),
            (strings(&["id"]), strings(&["x"]), "id", "value"),
            (strings(&["id"]), strings(&["x"]), "same", "same"),
        ] {
            assert!(base
                .melt(
                    Some(&ids),
                    Some(&values),
                    Some(var.into()),
                    Some(value.into())
                )
                .is_err());
        }
    }
}
#[test]
fn specialized_reshapes_reject_identity_loss_and_row_multiplication() {
    let nodes = NodesTable::from_base_table(table(vec![
        ("node_id", vec![V::Int(2)]),
        ("x", vec![V::Int(3)]),
        ("y", vec![V::Int(4)]),
        ("k", vec![V::Text("z".into())]),
    ]))
    .unwrap();
    assert!(nodes
        .melt(None, Some(&strings(&["x"])), None, None)
        .is_err());
    assert!(nodes
        .melt(
            Some(&strings(&["node_id"])),
            Some(&strings(&["x", "y"])),
            None,
            None
        )
        .is_err());
    let result = nodes
        .melt(
            Some(&strings(&["node_id"])),
            Some(&strings(&["x"])),
            None,
            None,
        )
        .unwrap();
    assert_eq!(result.node_ids().unwrap(), vec![2]);
    assert!(nodes
        .pivot_table(&strings(&["k"]), "k", "x", "sum")
        .is_err());
    assert_eq!(
        nodes
            .pivot_table(&strings(&["node_id"]), "k", "x", "sum")
            .unwrap()
            .node_ids()
            .unwrap(),
        vec![2]
    );
    assert_eq!(
        nodes
            .base_table()
            .melt(
                Some(&strings(&["node_id"])),
                Some(&strings(&["x", "y"])),
                None,
                None
            )
            .unwrap()
            .nrows(),
        2
    );
}
#[test]
fn edge_reshapes_require_ids_and_both_endpoints() {
    let edges = EdgesTable::from_base_table(table(vec![
        ("edge_id", vec![V::Int(9)]),
        ("source", vec![V::Int(2)]),
        ("target", vec![V::Int(3)]),
        ("x", vec![V::Int(4)]),
        ("k", vec![V::Text("z".into())]),
    ]))
    .unwrap();
    assert!(edges
        .melt(
            Some(&strings(&["edge_id"])),
            Some(&strings(&["x"])),
            None,
            None
        )
        .is_err());
    let ids = strings(&["edge_id", "source", "target"]);
    let result = edges
        .melt(Some(&ids), Some(&strings(&["x"])), None, None)
        .unwrap();
    assert_eq!(result.edge_ids().unwrap(), vec![9]);
    assert_eq!(result.targets().unwrap(), vec![3]);
    assert!(edges
        .pivot_table(&strings(&["edge_id"]), "k", "x", "sum")
        .is_err());
    assert_eq!(
        edges
            .pivot_table(&ids, "k", "x", "sum")
            .unwrap()
            .sources()
            .unwrap(),
        vec![2]
    );
}
#[test]
fn reshape_is_read_only_but_typed_filter_retains_direct_source_rows() {
    let mut graph = Graph::new();
    graph.add_node();
    graph.set_node_attr(0, "x".into(), V::Int(3)).unwrap();
    let view = graph.read_view().unwrap();
    let base = view.nodes_table().unwrap().into_base_table();
    let filtered = base.filter_by_value("x", &V::Int(3)).unwrap();
    assert_eq!(filtered.source_entity_ids(), Some(&[0][..]));
    let melted = base
        .melt(
            Some(&strings(&["node_id"])),
            Some(&strings(&["x"])),
            None,
            None,
        )
        .unwrap();
    assert!(melted.source_entity_ids().is_none());
    assert_eq!(melted.source_views().len(), 1);
}
