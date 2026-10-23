use groggy::storage::array::{ArrayArray, BaseArray};
use groggy::types::AttrValue;
#[test]
fn ragged_reductions_share_stability_and_explicit_empty_ieee_rules() {
    let a = ArrayArray::new(vec![
        BaseArray::new(vec![1e20, 1., -1e20]),
        BaseArray::new(vec![]),
        BaseArray::new(vec![f64::MAX, f64::MAX]),
        BaseArray::new(vec![0., 1e-300]),
        BaseArray::new(vec![f64::NAN]),
    ]);
    assert_eq!(&a.sum()[..2], &[1., 0.]);
    assert_eq!(a.mean()[2], f64::MAX);
    assert_eq!(a.std()[1], 0.);
    assert!((a.std()[3] / (1e-300 / 2f64.sqrt()) - 1.).abs() < 1e-15);
    for result in [a.sum(), a.mean(), a.min(), a.max(), a.std()] {
        assert!(result[4].is_nan());
    }
    assert_eq!(a.count(), vec![3, 0, 2, 2, 1]);
    assert_eq!(a.get(0).unwrap().as_slice(), &[1e20, 1., -1e20]);
}
#[test]
fn numeric_projection_keeps_named_keys_with_legacy_filtering() {
    let a = ArrayArray::with_named_keys(
        vec![BaseArray::new(vec![
            AttrValue::Float(1.25),
            AttrValue::Float(2.),
            AttrValue::Int(3),
            AttrValue::SmallInt(4),
            AttrValue::Bool(true),
            AttrValue::Null,
            AttrValue::Text("skip".into()),
        ])],
        vec!["a".into()],
        "kind",
    )
    .unwrap();
    let n = a.numeric_arrays().unwrap();
    assert_eq!(n.get(0).unwrap().as_slice(), &[1.25, 2., 3., 4., 1.]);
    assert_eq!(n.key_name().map(String::as_str), Some("kind"));
    assert_eq!(n.keys().unwrap(), &["a".to_string()]);
    assert_eq!(a.get(0).unwrap().len(), 7);
}
#[test]
fn table_aggregations_preserve_key_identity_and_check_narrowing_and_collisions() {
    let a =
        ArrayArray::with_named_keys(vec![BaseArray::new(vec![1., 2.])], vec!["a".into()], "kind")
            .unwrap();
    let value = a.std()[0];
    let t = a.to_table_with_aggregation("std", vec![value]).unwrap();
    assert_eq!(
        t.get_column("std").unwrap().get(0),
        Some(&AttrValue::Float64(value))
    );
    assert_eq!(
        t.get_column("kind").unwrap().get(0),
        Some(&AttrValue::Text("a".into()))
    );
    let t = a
        .aggregate(&[
            ("deviation".into(), "std".into()),
            ("n".into(), "count".into()),
        ])
        .unwrap();
    assert_eq!(
        t.get_column("deviation").unwrap().get(0),
        Some(&AttrValue::Float64(value))
    );
    assert!(a.to_table_with_aggregation("kind", vec![1.]).is_err());
    assert!(a.to_table_with_aggregation("x", vec![]).is_err());
    assert!(a.aggregate(&[("kind".into(), "sum".into())]).is_err());
    assert!(a
        .aggregate(&[("x".into(), "sum".into()), ("x".into(), "max".into())])
        .is_err());
    assert!(a.aggregate(&[("x".into(), "unknown".into())]).is_err());
    assert_eq!(
        a.to_table_with_aggregation("huge", vec![f64::MAX])
            .unwrap()
            .get_column("huge")
            .unwrap()
            .get(0),
        Some(&AttrValue::Float64(f64::MAX))
    );
    let stored = a
        .to_table_with_aggregation("ieee", vec![f64::INFINITY])
        .unwrap();
    assert_eq!(
        stored.get_column("ieee").unwrap().get(0),
        Some(&AttrValue::Float64(f64::INFINITY))
    );
}

#[test]
fn grouped_integer_reductions_keep_high_offset_differences_and_cancellation() {
    let arrays = ArrayArray::with_named_keys(
        vec![
            BaseArray::new(vec![
                AttrValue::Int(i64::MAX),
                AttrValue::Int(1),
                AttrValue::Int(-i64::MAX),
            ]),
            BaseArray::new(vec![
                AttrValue::Int(i64::MAX - 2),
                AttrValue::Int(i64::MAX - 1),
                AttrValue::Int(i64::MAX),
            ]),
            BaseArray::new(vec![
                AttrValue::Null,
                AttrValue::Text("skip".into()),
                AttrValue::Bool(true),
                AttrValue::SmallInt(3),
            ]),
            BaseArray::new(vec![]),
        ],
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        "kind",
    )
    .unwrap();
    assert_eq!(arrays.numeric_reduce("sum").unwrap()[0], 1.);
    assert_eq!(arrays.numeric_reduce("std").unwrap()[1], 1.);
    assert_eq!(arrays.numeric_reduce("mean").unwrap()[0], 1. / 3.);
    assert_eq!(arrays.numeric_counts(), vec![3, 3, 2, 0]);
    assert_eq!(arrays.numeric_reduce("mean").unwrap()[2], 2.);
    assert_eq!(arrays.numeric_reduce("std").unwrap()[3], 0.);
    let result = arrays
        .aggregate_numeric(&[("deviation".into(), "std".into())])
        .unwrap();
    assert_eq!(
        result.get_column("deviation").unwrap().get(1),
        Some(&AttrValue::Float64(1.))
    );
    assert!(result.get_column("kind").is_some());
    assert!(ArrayArray::new(Vec::<BaseArray<AttrValue>>::new())
        .numeric_reduce("unknown")
        .is_err());
}
