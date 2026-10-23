use groggy::{storage::array::BaseArray, types::AttrValue};
fn column(values: Vec<AttrValue>) -> BaseArray<AttrValue> {
    BaseArray::new(values)
}
fn f(value: AttrValue) -> f64 {
    match value {
        AttrValue::Float(v) => f64::from(v),
        AttrValue::Float64(v) => v,
        v => panic!("Expected float, got {v:?}"),
    }
}
#[test]
fn high_integer_and_mixed_correlations_and_covariance_keep_small_differences() {
    let m = i64::MAX;
    let a = column(vec![
        AttrValue::Int(m - 2),
        AttrValue::Int(m - 1),
        AttrValue::Int(m),
    ]);
    let b = column(vec![
        AttrValue::SmallInt(1),
        AttrValue::SmallInt(2),
        AttrValue::SmallInt(3),
    ]);
    for method in ["pearson", "spearman", "kendall"] {
        assert!((f(a.corr(&b, method).unwrap()) - 1.).abs() < 1e-14);
    }
    assert_eq!(f(a.cov(&b, 1).unwrap()), 1.);
    let mixed = column(vec![
        AttrValue::Int(m - 1),
        AttrValue::Int(m),
        AttrValue::Float(2f32.powi(63)),
    ]);
    assert_eq!(f(mixed.cov(&b, 1).unwrap()), 1.);
    assert!((f(mixed.corr(&b, "pearson").unwrap()) - 1.).abs() < 1e-14);
}
#[test]
fn quantile_midpoints_preserve_integer_residuals_and_validate_before_empty_return() {
    let a = column(vec![AttrValue::Int(i64::MIN), AttrValue::Int(i64::MAX)]);
    for interpolation in ["linear", "midpoint"] {
        assert_eq!(f(a.quantile(0.5, interpolation).unwrap()), -0.5);
    }
    let mixed = column(vec![
        AttrValue::Int(i64::MAX),
        AttrValue::Float(-2f32.powi(63)),
    ]);
    for interpolation in ["linear", "midpoint"] {
        assert_eq!(f(mixed.quantile(0.5, interpolation).unwrap()), -0.5);
    }
    let empty = column(vec![]);
    assert!(empty.quantile(0.5, "unknown").is_err());
    assert!(empty.quantile(f64::NAN, "linear").is_err());
    assert_eq!(empty.quantile(0.5, "linear").unwrap(), AttrValue::Null);
    let a = column(vec![
        AttrValue::SmallInt(2),
        AttrValue::Null,
        AttrValue::Text("ignored".into()),
        AttrValue::Int(8),
    ]);
    assert_eq!(f(a.quantile(0.5, "linear").unwrap()), 5.);
}
#[test]
fn paired_missing_values_shape_and_nonfinite_domains_are_explicit() {
    let x = column(vec![AttrValue::Int(1), AttrValue::Null, AttrValue::Int(3)]);
    let y = column(vec![
        AttrValue::SmallInt(2),
        AttrValue::SmallInt(99),
        AttrValue::SmallInt(6),
    ]);
    assert_eq!(f(x.cov(&y, 1).unwrap()), 4.);
    assert!((f(x.corr(&y, "pearson").unwrap()) - 1.).abs() < 1e-14);
    assert!(x.corr(&column(vec![]), "pearson").is_err());
    assert!(x.cov(&y, -1).is_err());
    for value in [f32::NAN, f32::INFINITY] {
        let a = column(vec![AttrValue::Float(value), AttrValue::Int(1)]);
        assert!(a.corr(&a, "spearman").is_err());
        assert!(a.quantile(0.5, "linear").is_err());
    }
    let huge = column(vec![
        AttrValue::Float64(f64::MAX),
        AttrValue::Float64(-f64::MAX),
    ]);
    assert!(huge.cov(&huge, 1).is_err());
    let constant = column(vec![AttrValue::Int(2); 3]);
    assert_eq!(f(constant.corr(&y, "pearson").unwrap()), 0.);
}
#[test]
fn legacy_kendall_tau_a_and_average_tie_ranks_are_preserved() {
    let a = column(vec![
        AttrValue::Int(1),
        AttrValue::Int(1),
        AttrValue::Int(2),
    ]);
    let b = column(vec![
        AttrValue::Int(1),
        AttrValue::Int(2),
        AttrValue::Int(3),
    ]);
    assert!((f(a.corr(&b, "kendall").unwrap()) - 2. / 3.).abs() < 1e-7);
    assert!((f(a.corr(&b, "spearman").unwrap()) - 3f64.sqrt() / 2.).abs() < 1e-7);
}
