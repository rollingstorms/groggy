use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    types::{CompactString, CompressedData},
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
fn values(t: &BaseTable, n: &str) -> Vec<V> {
    t.column(n).unwrap().data().clone()
}
#[test]
fn shared_typed_identity_drives_dedup_sets_and_both_group_apis() {
    let a = table(&[(
        "k",
        vec![
            V::Text("a".into()),
            V::CompactText(CompactString::new("a")),
            V::CompressedText(CompressedData::compress_text("a")),
            V::Int(2),
            V::SmallInt(2),
            V::Float64(-0.),
            V::Float64(0.),
            V::Float64(f64::NAN),
            V::Float64(f64::from_bits(0x7ff8000000000001)),
            V::Null,
            V::Null,
        ],
    )]);
    assert_eq!(a.drop_duplicates(None).unwrap().nrows(), 5);
    assert_eq!(a.union(&a).unwrap().nrows(), 5);
    assert_eq!(a.intersect(&a).unwrap().nrows(), 5);
    let groups = a.group_by(&["k".into()]).unwrap();
    assert_eq!(
        groups.iter().map(|g| g.nrows()).collect::<Vec<_>>(),
        vec![3, 2, 2, 2, 2]
    );
    let keyed = a.groupby(&["k"]).unwrap();
    assert_eq!(
        keyed.tables.iter().map(|g| g.nrows()).collect::<Vec<_>>(),
        vec![3, 2, 2, 2, 2]
    );
    assert_eq!(keyed.group_keys.unwrap()[0]["k"], V::Text("a".into()));
    let distinct = table(&[(
        "k",
        vec![
            V::Int(1),
            V::Float(1.),
            V::Float64(1.),
            V::Bool(true),
            V::Text("1".into()),
            V::Json("1".into()),
        ],
    )]);
    assert_eq!(distinct.drop_duplicates(None).unwrap().nrows(), 6);
}
#[test]
fn delimited_text_cannot_collide_in_composite_keys() {
    let a = table(&[
        ("x", vec![V::Text("a|t:b".into()), V::Text("a".into())]),
        ("y", vec![V::Text("c".into()), V::Text("b|t:c".into())]),
    ]);
    assert_eq!(a.drop_duplicates(None).unwrap().nrows(), 2);
    assert_eq!(a.union(&a).unwrap().nrows(), 2);
    assert_eq!(a.groupby(&["x", "y"]).unwrap().len(), 2);
    assert_eq!(a.drop_duplicates(Some(&[])).unwrap().nrows(), 1);
}
#[test]
fn set_operations_match_nested_equality_oracles_and_keep_first_rows() {
    let keys = [V::Null, V::Int(1), V::SmallInt(1), V::Int(2)];
    for a_len in 0..6 {
        for b_len in 0..6 {
            for offset in 0..4 {
                let ak = (0..a_len).map(|i| keys[i % 4].clone()).collect::<Vec<_>>();
                let bk = (0..b_len)
                    .map(|i| keys[(i + offset) % 4].clone())
                    .collect::<Vec<_>>();
                let a = table(&[("k", ak.clone())]);
                let b = table(&[("k", bk.clone())]);
                let mut union = Vec::new();
                for k in ak.iter().chain(&bk) {
                    if !union.contains(k) {
                        union.push(k.clone());
                    }
                }
                let mut intersection = Vec::new();
                for k in &ak {
                    if bk.contains(k) && !intersection.contains(k) {
                        intersection.push(k.clone());
                    }
                }
                assert_eq!(values(&a.union(&b).unwrap(), "k"), union);
                assert_eq!(values(&a.intersect(&b).unwrap(), "k"), intersection);
                assert_eq!(
                    values(&a.stack(&b).unwrap(), "k"),
                    ak.iter().chain(&bk).cloned().collect::<Vec<_>>()
                );
            }
        }
    }
}
#[test]
fn zero_column_rows_empty_inputs_and_schema_validation_are_explicit() {
    let a = table(&[("k", vec![V::Int(1), V::Int(2)])]);
    let z = a.select(&[]).unwrap();
    let empty = z.head(0);
    assert_eq!(z.stack(&z).unwrap().nrows(), 4);
    assert_eq!(z.union(&z).unwrap().nrows(), 1);
    assert_eq!(z.intersect(&empty).unwrap().nrows(), 0);
    assert_eq!(z.drop_duplicates(None).unwrap().nrows(), 1);
    assert_eq!(z.groupby(&[]).unwrap().tables[0].nrows(), 2);
    assert_eq!(empty.groupby(&[]).unwrap().len(), 0);
    assert_eq!(empty.group_by(&[]).unwrap().len(), 1);
    let reordered = table(&[("b", vec![V::Int(1)]), ("a", vec![V::Int(2)])]);
    let other = table(&[("a", vec![V::Int(1)]), ("b", vec![V::Int(2)])]);
    assert!(reordered.union(&other).is_err());
    assert!(a.stack(&other).is_err());
    for keys in [vec!["missing"], vec!["k", "k"]] {
        assert!(a.groupby(&keys).is_err());
        assert!(a
            .group_by(&keys.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .is_err());
        assert!(a
            .drop_duplicates(Some(
                &keys.iter().map(|s| s.to_string()).collect::<Vec<_>>()
            ))
            .is_err());
    }
}
#[test]
fn concat_preserves_column_order_padding_and_rejects_collisions() {
    let a = table(&[("b", vec![V::Int(1)]), ("a", vec![V::Int(2)])]);
    let b = table(&[
        ("a", vec![V::Int(3), V::Int(4)]),
        ("c", vec![V::Int(5), V::Int(6)]),
    ]);
    let c = a.concat(&b).unwrap();
    assert_eq!(c.column_names(), &["b", "a", "a_y", "c"]);
    assert_eq!(values(&c, "b"), vec![V::Int(1), V::Null]);
    assert_eq!(values(&c, "a_y"), vec![V::Int(3), V::Int(4)]);
    let conflict = table(&[("a_y", vec![V::Int(9)]), ("a", vec![V::Int(10)])]);
    assert!(a.concat(&conflict).is_err());
    assert_eq!(a.nrows(), 1);
    assert_eq!(
        a.select(&[])
            .unwrap()
            .concat(&b.select(&[]).unwrap())
            .unwrap()
            .nrows(),
        2
    );
}
#[test]
fn grouping_aggregates_have_stable_columns_and_reject_bad_empty_specs() {
    let a = table(&[
        ("k", vec![V::Int(2), V::Int(1), V::Int(2)]),
        ("z", vec![V::Int(1), V::Int(2), V::Int(3)]),
        ("a", vec![V::Int(4), V::Int(5), V::Int(6)]),
    ]);
    let specs = HashMap::from([("z".into(), "sum".into()), ("a".into(), "count".into())]);
    for _ in 0..5 {
        let grouped = a.group_by_agg(&["k".into()], specs.clone()).unwrap();
        assert_eq!(grouped.column_names(), &["k", "a", "z"]);
        assert_eq!(values(&grouped, "k"), vec![V::Int(2), V::Int(1)]);
        assert_eq!(values(&grouped, "a"), vec![V::Int(2), V::Int(1)]);
        assert_eq!(
            a.aggregate(specs.clone()).unwrap().column_names(),
            &["a", "z"]
        );
    }
    let empty = a.head(0);
    for (col, op) in [("k", "sum"), ("z", "bad"), ("missing", "sum")] {
        assert!(empty
            .group_by_agg(&["k".into()], HashMap::from([(col.into(), op.into())]))
            .is_err());
    }
}
#[test]
fn group_selections_retain_write_identity_while_combination_and_aggregation_do_not() {
    let mut g = Graph::new_publishing();
    g.insert_nodes(3).unwrap();
    g.set_node_attrs(HashMap::from([(
        "k".into(),
        vec![(0, V::Int(2)), (1, V::Int(1)), (2, V::Int(2))],
    )]))
    .unwrap();
    let t = g.read_view().unwrap().nodes_table().unwrap().into_base();
    let groups = t.groupby(&["k"]).unwrap();
    assert_eq!(groups.tables[0].source_entity_ids(), Some(&[0, 2][..]));
    let dedup = t.drop_duplicates(Some(&["k".into()])).unwrap();
    assert_eq!(dedup.source_entity_ids(), Some(&[0, 1][..]));
    let edit = groups.tables[0]
        .with_column(
            "k".into(),
            BaseArray::from_attr_values(vec![V::Int(9), V::Int(9)]),
        )
        .unwrap();
    edit.update_source_graph(&mut g, &["k".into()], None, None)
        .unwrap();
    assert_eq!(g.get_node_attr(2, &"k".into()).unwrap(), Some(V::Int(9)));
    for derived in [
        t.union(&t).unwrap(),
        t.intersect(&t).unwrap(),
        t.stack(&t).unwrap(),
        t.concat(&t).unwrap(),
        t.group_by_agg(
            &["k".into()],
            HashMap::from([("node_id".into(), "count".into())]),
        )
        .unwrap(),
    ] {
        assert_eq!(derived.source_views().len(), 1);
        assert_eq!(derived.source_entity_ids(), None);
        assert!(derived
            .update_source_graph(&mut g, &["k".into()], None, None)
            .is_err());
    }
    assert_eq!(
        t.source_views()[0].node_attr(2, &"k".into()).unwrap(),
        Some(V::Int(2))
    );
}
