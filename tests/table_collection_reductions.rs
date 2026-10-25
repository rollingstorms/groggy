use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table, TableArray},
    },
    types::CompactString,
    AttrValue as V, Graph,
};
use std::collections::HashMap;
fn table(columns: &[(&str, Vec<V>)]) -> BaseTable {
    BaseTable::with_column_order(
        columns
            .iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v.clone())))
            .collect(),
        columns.iter().map(|(n, _)| n.to_string()).collect(),
    )
    .unwrap()
}
fn col(t: &BaseTable, n: &str) -> Vec<V> {
    t.column(n).unwrap().data().clone()
}
fn spec(name: &str, op: &str) -> HashMap<String, String> {
    HashMap::from([(name.into(), op.into())])
}
#[test]
fn grouped_defaults_preserve_keys_once_and_skip_text_without_reducing_keys() {
    let t = table(&[
        (
            "k",
            vec![
                V::Text("b".into()),
                V::Text("a".into()),
                V::Text("b".into()),
            ],
        ),
        ("value", vec![V::Int(1), V::Int(4), V::Int(2)]),
        ("text", vec![V::Text("x".into()); 3]),
    ]);
    let t = t
        .with_column(
            "flag".into(),
            BaseArray::from_attr_values(vec![V::Bool(false), V::Bool(true), V::Bool(true)]),
        )
        .unwrap();
    let groups = t.groupby(&["k"]).unwrap();
    let sum = groups.sum().unwrap();
    assert_eq!(sum.column_names(), &["k", "value"]);
    assert_eq!(
        col(&sum, "k"),
        vec![V::Text("b".into()), V::Text("a".into())]
    );
    assert_eq!(col(&sum, "value"), vec![V::Int(3), V::Int(4)]);
    assert_eq!(
        col(&groups.mean().unwrap(), "value"),
        vec![V::Float64(1.5), V::Float64(4.)]
    );
    assert_eq!(
        col(&groups.count().unwrap(), "text"),
        vec![V::Int(2), V::Int(1)]
    );
    assert_eq!(groups.count().unwrap().nrows(), 2);
}
#[test]
fn ungrouped_reductions_stream_all_observations_and_keep_exact_integer_cancellation() {
    let a = TableArray::from_tables(vec![
        table(&[("x", vec![V::Int(i64::MAX)])]),
        table(&[("x", vec![V::Int(1), V::Int(-i64::MAX)])]),
    ]);
    assert_eq!(col(&a.sum().unwrap(), "x"), vec![V::Int(1)]);
    assert_eq!(col(&a.mean().unwrap(), "x"), vec![V::Float64(1. / 3.)]);
    assert_eq!(col(&a.count().unwrap(), "x"), vec![V::Int(3)]);
    let b = TableArray::from_tables(vec![table(&[("x", vec![V::Float64(f64::MAX)])]); 2]);
    assert!(b.sum().is_err());
    assert_eq!(col(&b.mean().unwrap(), "x"), vec![V::Float64(f64::MAX)]);
    let precise = TableArray::from_tables(vec![
        table(&[("x", vec![V::Int(9_007_199_254_740_993)])]),
        table(&[("x", vec![V::Float64(9_007_199_254_740_992.)])]),
    ]);
    assert_eq!(
        col(&precise.min().unwrap(), "x"),
        vec![V::Float64(9_007_199_254_740_992.)]
    );
    assert_eq!(
        col(&precise.max().unwrap(), "x"),
        vec![V::Int(9_007_199_254_740_993)]
    );
}
#[test]
fn explicit_specs_nulls_empty_groups_unique_counts_and_boolean_validation() {
    let a = table(&[
        ("x", vec![V::Null, V::Null]),
        ("b", vec![V::Bool(false), V::Bool(true)]),
        (
            "t",
            vec![V::Text("a".into()), V::CompactText(CompactString::new("a"))],
        ),
    ]);
    let g = TableArray::from_group_columns(vec![a, a_empty()], &[]).unwrap();
    assert_eq!(
        col(&g.agg(spec("x", "sum")).unwrap(), "x"),
        vec![V::Null, V::Null]
    );
    assert_eq!(
        col(&g.agg(spec("x", "count")).unwrap(), "x"),
        vec![V::Int(0), V::Int(0)]
    );
    assert_eq!(
        col(&g.agg(spec("t", "nunique")).unwrap(), "t"),
        vec![V::Int(1), V::Int(0)]
    );
    assert_eq!(col(&g.all().unwrap(), "b"), vec![V::Bool(false), V::Null]);
    let invalid = TableArray::from_tables(vec![table(&[(
        "b",
        vec![V::Bool(false), V::Text("bad".into())],
    )])]);
    assert!(invalid.all().is_err());
    assert!(invalid.agg(spec("b", "all")).is_err());
    for op in ["sum", "mean", "min", "max"] {
        let bad = TableArray::from_tables(vec![table(&[(
            "x",
            vec![V::Float64(1.), V::Float64(f64::INFINITY)],
        )])]);
        assert!(bad.agg(spec("x", op)).is_err());
    }
}
fn a_empty() -> BaseTable {
    table(&[("x", vec![]), ("b", vec![]), ("t", vec![])])
}
#[test]
fn invalid_group_metadata_and_specs_reject_without_partial_output() {
    let t = table(&[
        ("k", vec![V::Int(1), V::Int(2)]),
        ("x", vec![V::Int(3), V::Int(4)]),
    ]);
    assert!(TableArray::from_group_columns(vec![t.clone()], &["k".into()]).is_err());
    assert!(TableArray::from_group_columns(vec![t.clone()], &["missing".into()]).is_err());
    let wrong_count = TableArray::from_tables_with_keys(vec![t.clone()], vec![]);
    assert!(wrong_count.sum().is_err());
    assert!(wrong_count.concat().is_err());
    let wrong_schemas = TableArray::from_tables_with_keys(
        vec![t.clone(), t.clone()],
        vec![HashMap::from([("a".into(), V::Int(1))]), HashMap::new()],
    );
    assert!(wrong_schemas.count().is_err());
    let groups = t.groupby(&["k"]).unwrap();
    for (n, op) in [("k", "sum"), ("missing", "sum"), ("x", "bad")] {
        assert!(groups.agg(spec(n, op)).is_err());
    }
    let empty = TableArray::new();
    assert!(empty.agg(spec("x", "bad")).is_err());
    assert_eq!(empty.agg(spec("x", "sum")).unwrap().nrows(), 0);
}
#[test]
fn heterogeneous_flatten_is_lossless_ordered_padded_and_source_aware() {
    let a = table(&[("a", vec![V::Int(1)])]);
    let b = table(&[("b", vec![V::Int(2), V::Int(3)])]);
    let array = TableArray::from_tables(vec![a, b]);
    let flat = array.concat().unwrap();
    assert_eq!(flat.column_names(), &["a", "b"]);
    assert_eq!(col(&flat, "a"), vec![V::Int(1), V::Null, V::Null]);
    assert_eq!(col(&flat, "b"), vec![V::Null, V::Int(2), V::Int(3)]);
    assert_eq!(col(&array.sum().unwrap(), "a"), vec![V::Int(1)]);
    assert_eq!(
        col(&array.agg(spec("b", "count")).unwrap(), "b"),
        vec![V::Int(2)]
    );
    let zero = TableArray::from_tables(vec![flat.select(&[]).unwrap(), flat.select(&[]).unwrap()]);
    assert_eq!(zero.concat().unwrap().nrows(), 6);
    assert_eq!(zero.count().unwrap().nrows(), 1);
    assert_eq!(zero.agg(HashMap::new()).unwrap().nrows(), 1);
}
#[test]
fn aggregate_and_flatten_outputs_retain_distinct_source_pins_without_write_identity() {
    let mut owners = [Graph::new_publishing(), Graph::new_publishing()];
    let mut tables = Vec::new();
    for (i, g) in owners.iter_mut().enumerate() {
        g.insert_nodes(2).unwrap();
        g.set_node_attrs(HashMap::from([(
            "x".into(),
            vec![(0, V::Int(i as i64)), (1, V::Int(1))],
        )]))
        .unwrap();
        tables.push(g.read_view().unwrap().nodes_table().unwrap().into_base());
    }
    let collection = TableArray::from_tables(tables);
    let sum = collection.agg(spec("x", "sum")).unwrap();
    let flat = collection.concat().unwrap();
    for result in [&sum, &flat] {
        assert_eq!(result.source_views().len(), 2);
        assert_eq!(result.source_entity_ids(), None);
    }
    owners[0].set_node_attr(0, "x".into(), V::Int(99)).unwrap();
    assert_eq!(
        sum.source_views()[0].node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(0))
    );
}
#[test]
fn grouped_reductions_match_direct_row_oracles_with_stable_key_order() {
    for n in 1..30 {
        let t = table(&[
            ("k", (0..n).map(|i| V::Int((i % 3) as i64)).collect()),
            ("x", (0..n).map(|i| V::Int(i as i64 - 10)).collect()),
        ]);
        let groups = t.groupby(&["k"]).unwrap();
        let sum = groups.sum().unwrap();
        let mean = groups.mean().unwrap();
        let min = groups.min().unwrap();
        let max = groups.max().unwrap();
        let count = groups.count().unwrap();
        for k in 0..n.min(3) {
            let values = (0..n)
                .filter(|i| i % 3 == k)
                .map(|i| i as i64 - 10)
                .collect::<Vec<_>>();
            assert_eq!(col(&sum, "x")[k], V::Int(values.iter().sum()));
            assert_eq!(
                col(&mean, "x")[k],
                V::Float64(values.iter().sum::<i64>() as f64 / values.len() as f64)
            );
            assert_eq!(col(&min, "x")[k], V::Int(*values.iter().min().unwrap()));
            assert_eq!(col(&max, "x")[k], V::Int(*values.iter().max().unwrap()));
            assert_eq!(col(&count, "x")[k], V::Int(values.len() as i64));
        }
    }
}
