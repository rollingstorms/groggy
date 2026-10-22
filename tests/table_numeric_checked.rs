use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue,
};
use std::collections::HashMap;
fn table(columns: Vec<(&str, Vec<AttrValue>)>) -> BaseTable {
    BaseTable::from_columns(
        columns
            .into_iter()
            .map(|(k, v)| (k.into(), BaseArray::new(v)))
            .collect(),
    )
    .unwrap()
}
fn aggregate(table: &BaseTable, operation: &str) -> AttrValue {
    table
        .aggregate(HashMap::from([("x".into(), operation.into())]))
        .unwrap()
        .column("x")
        .unwrap()
        .data()[0]
        .clone()
}
#[test]
fn aggregate_preserves_integer_sums_compact_values_and_missing_policy() {
    let t = table(vec![(
        "x",
        vec![
            AttrValue::Int(i64::MAX),
            AttrValue::SmallInt(1),
            AttrValue::Int(-i64::MAX),
            AttrValue::Null,
            AttrValue::Text("skip".into()),
        ],
    )]);
    assert_eq!(aggregate(&t, "sum"), AttrValue::Int(1));
    assert_eq!(aggregate(&t, "count"), AttrValue::Int(5));
    assert_eq!(aggregate(&t, "mean"), AttrValue::Float(1. / 3.));
    let t = table(vec![(
        "x",
        vec![
            AttrValue::SmallInt(3),
            AttrValue::Int(1),
            AttrValue::Float(2.),
        ],
    )]);
    assert_eq!(aggregate(&t, "min"), AttrValue::Int(1));
    assert_eq!(aggregate(&t, "max"), AttrValue::SmallInt(3));
    assert_eq!(aggregate(&t, "sum"), AttrValue::Float(6.));
}
#[test]
fn aggregates_reject_integer_float_overflow_and_nonfinite_values() {
    for values in [
        vec![AttrValue::Int(i64::MAX), AttrValue::Int(1)],
        vec![AttrValue::Float(f32::MAX); 2],
        vec![AttrValue::Float(f32::NAN)],
    ] {
        let t = table(vec![("x", values)]);
        assert!(t
            .aggregate(HashMap::from([("x".into(), "sum".into())]))
            .is_err());
    }
}
#[test]
fn sorting_uses_exact_mixed_numeric_order_and_lexical_text_order() {
    let t = table(vec![(
        "x",
        vec![
            AttrValue::Float(2f32.powi(63)),
            AttrValue::Int(i64::MAX),
            AttrValue::SmallInt(1),
        ],
    )]);
    assert_eq!(
        t.nsmallest(3, "x").unwrap().column("x").unwrap().data(),
        &vec![
            AttrValue::SmallInt(1),
            AttrValue::Int(i64::MAX),
            AttrValue::Float(2f32.powi(63))
        ]
    );
    let t = table(vec![(
        "x",
        vec![AttrValue::Text("z".into()), AttrValue::Text("a".into())],
    )]);
    assert_eq!(
        t.sort_values(vec!["x".into()], vec![true])
            .unwrap()
            .column("x")
            .unwrap()
            .data(),
        &vec![AttrValue::Text("a".into()), AttrValue::Text("z".into())]
    );
}
#[test]
fn pivot_keys_match_identity_and_text_facts_do_not_merge() {
    let t = table(vec![
        ("g", vec![AttrValue::Int(1); 2]),
        (
            "key",
            vec![AttrValue::Text("a".into()), AttrValue::Text("b".into())],
        ),
        ("x", vec![AttrValue::SmallInt(2), AttrValue::SmallInt(7)]),
    ]);
    let p = t.pivot_table(&["g".into()], "key", "x", "sum").unwrap();
    assert_eq!(
        p.column("key_a").unwrap().data(),
        &vec![AttrValue::Float(2.)]
    );
    assert_eq!(
        p.column("key_b").unwrap().data(),
        &vec![AttrValue::Float(7.)]
    );
    let t = table(vec![
        ("g", vec![AttrValue::Int(1); 2]),
        (
            "key",
            vec![AttrValue::Int(i64::MAX), AttrValue::Float(2f32.powi(63))],
        ),
        ("x", vec![AttrValue::Int(2), AttrValue::Int(7)]),
    ]);
    let p = t.pivot_table(&["g".into()], "key", "x", "sum").unwrap();
    let mut sums: Vec<_> = p
        .columns()
        .iter()
        .filter(|(k, _)| k.starts_with("key_"))
        .map(|(_, v)| v.data()[0].clone())
        .collect();
    sums.sort();
    assert_eq!(sums, vec![AttrValue::Float(2.), AttrValue::Float(7.)]);
}
#[test]
fn pivot_rejects_ambiguous_output_names_and_empty_invalid_aggregation() {
    let t = table(vec![
        ("g", vec![AttrValue::Int(1); 2]),
        ("key", vec![AttrValue::Int(1), AttrValue::Float(1.)]),
        ("x", vec![AttrValue::Int(2), AttrValue::Int(7)]),
    ]);
    assert!(t.pivot_table(&["g".into()], "key", "x", "sum").is_err());
    let t = table(vec![("g", vec![]), ("key", vec![]), ("x", vec![])]);
    assert!(t.pivot_table(&["g".into()], "key", "x", "bad").is_err());
}
#[test]
fn describe_retains_population_spread_and_small_cancellation() {
    let t = table(vec![(
        "x",
        vec![
            AttrValue::Int(i64::MAX - 2),
            AttrValue::Int(i64::MAX - 1),
            AttrValue::Int(i64::MAX),
        ],
    )]);
    let d = t.describe().unwrap();
    assert_eq!(
        d.column("x").unwrap().data()[2],
        AttrValue::Float((2f32 / 3.).sqrt())
    );
    let t = table(vec![(
        "x",
        vec![AttrValue::Int(i64::MAX), AttrValue::Float(-2f32.powi(63))],
    )]);
    let d = t.describe().unwrap();
    let values = d.column("x").unwrap().data();
    assert_eq!(values[0], AttrValue::Float(2.));
    assert_eq!(values[1], AttrValue::Float(-0.5));
    assert_eq!(values[5], AttrValue::Float(-0.5));
    let t = table(vec![("x", vec![AttrValue::Float(f32::INFINITY)])]);
    assert!(t.describe().is_err());
}
