use groggy::{
    storage::{
        array::{ArrayArray, BaseArray, NumArray},
        numeric::scalar::ConversionPolicy,
        table::{BaseTable, TableArray},
    },
    AttrValue as V,
};
use std::collections::HashMap;
fn value(v: V) -> f64 {
    match v {
        V::Float64(v) => v,
        _ => panic!("expected f64 result: {v:?}"),
    }
}
fn column(values: Vec<V>) -> BaseArray<V> {
    BaseArray::new(values)
}
fn table(values: Vec<V>) -> BaseTable {
    BaseTable::from_columns(HashMap::from([("x".into(), column(values))])).unwrap()
}
#[test]
fn statistics_windows_and_numeric_materialization_keep_f64_precision() {
    let x = 1. + 2f64.powi(-40);
    let y = x + 2f64.powi(-41);
    let a = column(vec![V::Float64(x), V::Float64(y)]);
    assert_eq!(value(a.sum().unwrap()), x + y);
    assert_eq!(value(a.quantile(0.5, "linear").unwrap()), (x + y) / 2.);
    assert_eq!(value(a.cumsum().unwrap().data()[1].clone()), x + y);
    assert_eq!(
        value(a.rolling(2, "mean").unwrap().data()[1].clone()),
        (x + y) / 2.
    );
    let attrs = NumArray::new(vec![x, 1e300])
        .to_attributes_with_policy(ConversionPolicy::Exact)
        .unwrap();
    assert_eq!(attrs, vec![V::Float64(x), V::Float64(1e300)]);
    assert!(matches!(
        NumArray::new(vec![0.5_f32]).to_attributes().unwrap()[0],
        V::Float(_)
    ));
    let aggregate = ArrayArray::new(vec![a])
        .aggregate_numeric(&[("mean".into(), "mean".into())])
        .unwrap();
    assert_eq!(
        value(aggregate.get_column("mean").unwrap().data()[0].clone()),
        (x + y) / 2.
    );
}
#[test]
fn representable_wide_means_survive_sum_overflow_and_actual_overflow_still_fails() {
    let a = column(vec![V::Float64(f64::MAX); 2]);
    assert!(a.sum().is_err());
    assert_eq!(a.mean().unwrap(), f64::MAX);
    assert_eq!(
        a.expanding("mean").unwrap().data(),
        &vec![V::Float64(f64::MAX); 2]
    );
    let a = column(vec![
        V::Float64(f64::MAX),
        V::Float64(f64::MAX),
        V::Float64(-f64::MAX),
        V::Float64(-f64::MAX),
        V::Int(i64::MAX),
        V::Float64(-2f64.powi(63)),
        V::Int(2),
    ]);
    assert_eq!(value(a.sum().unwrap()), 1.);
    assert_eq!(a.mean().unwrap(), 1. / 7.);
    let a = column(vec![V::Float64(-1e154), V::Float64(1e154)]);
    assert!(a.var().is_err()); // Sample variance exceeds f64, standard deviation does not.
    assert!((a.std().unwrap() / 1e154 - 2f64.sqrt()).abs() < 1e-14);
    assert!((value(a.cov(&a, 0).unwrap()) / 1e308 - 1.).abs() < 1e-14);
    assert!((value(a.expanding("std").unwrap().data()[1].clone()) / 1e154 - 1.).abs() < 1e-14);
}
#[test]
fn cross_table_reductions_stream_values_and_preserve_cancellation_across_chunks() {
    let tables = TableArray::from_tables(vec![
        table(vec![V::Int(i64::MAX)]),
        table(vec![V::Int(1)]),
        table(vec![V::Int(-i64::MAX)]),
    ]);
    assert_eq!(tables.sum_column("x").unwrap(), V::Int(1));
    assert_eq!(tables.mean_column("x").unwrap(), 1. / 3.);
    let tables = TableArray::from_tables(vec![
        table(vec![V::Float64(f64::MAX)]),
        table(vec![V::Float64(f64::MAX)]),
    ]);
    assert!(tables.sum_column("x").is_err());
    assert_eq!(tables.mean_column("x").unwrap(), f64::MAX);
    let tables = TableArray::from_tables(vec![
        table(vec![V::Float64(-1e154)]),
        table(vec![V::Float64(1e154)]),
    ]);
    assert!((tables.std_column("x").unwrap() / 1e154 - 2f64.sqrt()).abs() < 1e-14);
}

#[test]
fn streaming_moments_rescale_before_root_and_never_hide_nonfinite_updates() {
    let a = column(vec![V::Float64(-1e300), V::Float64(1e300)]);
    assert!((value(a.expanding("std").unwrap().data()[1].clone()) / 1e300 - 1.).abs() < 1e-14);
    let a = column(vec![V::Float64(-f64::MAX), V::Float64(f64::MAX)]);
    assert!(a.expanding("std").is_err());
}
