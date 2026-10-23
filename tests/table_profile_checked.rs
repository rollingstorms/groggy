use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue,
};
fn profile(values: Vec<AttrValue>) -> Result<BaseTable, groggy::GraphError> {
    BaseTable::from_columns(std::collections::HashMap::from([(
        "x".into(),
        BaseArray::new(values),
    )]))?
    .profile()
}
fn cell(table: &BaseTable, column: &str) -> AttrValue {
    table.column(column).unwrap().data()[0].clone()
}
fn float(table: &BaseTable, column: &str) -> f64 {
    match cell(table, column) {
        AttrValue::Float64(value) => value,
        other => panic!("expected f64 {column}: {other:?}"),
    }
}
#[test]
fn profile_retains_precision_population_spread_and_source_rank_values() {
    let x = 1.0 + 2f64.powi(-40);
    let p = profile(vec![AttrValue::Float64(x), AttrValue::Float64(x)]).unwrap();
    assert_eq!(cell(&p, "dtype"), AttrValue::Text("float64".into()));
    assert_eq!(float(&p, "mean"), x);
    assert_eq!(float(&p, "median"), x);
    assert_eq!(float(&p, "std"), 0.0);
    let p = profile(vec![
        AttrValue::Int(i64::MAX - 2),
        AttrValue::Int(i64::MAX - 1),
        AttrValue::Int(i64::MAX),
    ])
    .unwrap();
    assert!((float(&p, "std") - (2.0f64 / 3.0).sqrt()).abs() < 1e-15);
    assert_eq!(cell(&p, "min"), AttrValue::Int(i64::MAX - 2));
    assert_eq!(cell(&p, "max"), AttrValue::Int(i64::MAX));
    assert_eq!(cell(&p, "q25"), AttrValue::Int(i64::MAX - 2));
    assert_eq!(cell(&p, "q75"), AttrValue::Int(i64::MAX));
    for n in 1..=12 {
        let p = profile((0..n).map(AttrValue::Int).collect()).unwrap();
        assert_eq!(cell(&p, "q25"), AttrValue::Int(n / 4));
        assert_eq!(cell(&p, "q75"), AttrValue::Int(3 * n / 4));
    }
}
#[test]
fn profile_uses_checked_extreme_statistics_and_rejects_nonfinite_values() {
    let p = profile(vec![AttrValue::Float64(f64::MAX); 2]).unwrap();
    assert_eq!(float(&p, "mean"), f64::MAX);
    assert_eq!(float(&p, "median"), f64::MAX);
    assert_eq!(float(&p, "std"), 0.0);
    let p = profile(vec![AttrValue::Float64(-1e300), AttrValue::Float64(1e300)]).unwrap();
    assert!((float(&p, "std") / 1e300 - 1.0).abs() < 1e-15);
    assert_eq!(float(&p, "median"), 0.0);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(profile(vec![AttrValue::Float64(value)]).is_err());
    }
}
#[test]
fn profile_counts_typed_values_once_and_defines_empty_ratios_as_absent() {
    let p = profile(vec![
        AttrValue::Text("a".into()),
        AttrValue::Text("a".into()),
        AttrValue::Null,
    ])
    .unwrap();
    assert_eq!(cell(&p, "unique_count"), AttrValue::Int(2));
    assert_eq!(float(&p, "cardinality"), 2.0 / 3.0);
    assert_eq!(cell(&p, "mean"), AttrValue::Null);
    let p = profile(vec![]).unwrap();
    for key in [
        "mean",
        "std",
        "min",
        "max",
        "null_percentage",
        "cardinality",
    ] {
        assert_eq!(cell(&p, key), AttrValue::Null);
    }
    assert_eq!(cell(&p, "unique_count"), AttrValue::Int(0));
    let p = profile(vec![AttrValue::Float64Vec(vec![1.]), AttrValue::Bool(true)]).unwrap();
    assert_eq!(cell(&p, "dtype"), AttrValue::Text("boolean".into()));
    assert_eq!(cell(&p, "mean"), AttrValue::Null);
}
