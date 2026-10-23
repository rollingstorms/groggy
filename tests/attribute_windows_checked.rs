use groggy::{storage::array::BaseArray, AttrValue};
fn column(values: Vec<AttrValue>) -> BaseArray<AttrValue> {
    BaseArray::new(values)
}
fn floats(values: &[Option<f64>]) -> Vec<AttrValue> {
    values
        .iter()
        .map(|v| v.map_or(AttrValue::Null, AttrValue::Float64))
        .collect()
}
#[test]
fn windows_preserve_high_integer_residuals_and_population_std() {
    let m = i64::MAX;
    let a = column(vec![
        AttrValue::Int(m - 2),
        AttrValue::Int(m - 1),
        AttrValue::Int(m),
    ]);
    assert_eq!(
        a.expanding("std").unwrap().data(),
        &floats(&[Some(0.), Some(0.5), Some((2f64 / 3.).sqrt())])
    );
    assert_eq!(
        a.rolling(2, "std").unwrap().data(),
        &floats(&[None, Some(0.5), Some(0.5)])
    );
    let mixed = column(vec![AttrValue::Int(m), AttrValue::Float(2f32.powi(63))]);
    assert_eq!(
        mixed.expanding("std").unwrap().data(),
        &floats(&[Some(0.), Some(0.5)])
    );
}
#[test]
fn cumulative_exact_sum_survives_mixed_cancellation_and_compact_values() {
    let a = column(vec![
        AttrValue::Null,
        AttrValue::Int(i64::MAX),
        AttrValue::Float(-2f32.powi(63)),
        AttrValue::SmallInt(3),
        AttrValue::Null,
    ]);
    let sum = a.cumsum().unwrap();
    assert_eq!(sum.data()[0], AttrValue::Float64(0.));
    assert_eq!(&sum.data()[2..], &floats(&[Some(-1.), Some(2.), Some(2.)]));
    let expanding = a.expanding("mean").unwrap();
    assert_eq!(expanding.data()[0], AttrValue::Null);
    assert_eq!(expanding.data()[2], AttrValue::Float64(-0.5));
    let a = column(vec![
        AttrValue::Null,
        AttrValue::SmallInt(3),
        AttrValue::SmallInt(1),
        AttrValue::Null,
    ]);
    assert_eq!(
        a.cummin().unwrap().data(),
        &floats(&[None, Some(3.), Some(1.), Some(1.)])
    );
    assert_eq!(
        a.cummax().unwrap().data(),
        &floats(&[None, Some(3.), Some(3.), Some(3.)])
    );
    assert_eq!(
        a.rolling(2, "sum").unwrap().data(),
        &floats(&[None, Some(3.), Some(4.), Some(1.)])
    );
}
#[test]
fn windows_validate_empty_domains_and_narrowed_overflow() {
    let empty = column(vec![]);
    assert!(empty.expanding("bad").is_err());
    assert!(empty.rolling(1, "bad").is_err());
    assert!(empty.rolling(0, "sum").is_err());
    let invalid = column(vec![AttrValue::Float(f32::NAN)]);
    assert!(invalid.rolling(2, "sum").is_err());
    assert!(invalid.cumsum().is_err());
    let huge = column(vec![AttrValue::Float64(f64::MAX); 2]);
    assert!(huge.cumsum().is_err());
    assert_eq!(
        huge.expanding("mean").unwrap().data(),
        &vec![AttrValue::Float64(f64::MAX); 2]
    );
}
#[test]
fn shifts_outside_length_and_minimum_period_are_total() {
    for a in [
        column(vec![]),
        column(vec![AttrValue::Int(1), AttrValue::Int(2)]),
    ] {
        for period in [i32::MIN, -10, 10, i32::MAX] {
            assert_eq!(
                a.shift(period, None).unwrap().data(),
                &vec![AttrValue::Null; a.len()]
            );
        }
    }
    let a = column(vec![AttrValue::Int(1), AttrValue::Int(2)]);
    assert_eq!(
        a.shift(-1, None).unwrap().data(),
        &vec![AttrValue::Int(2), AttrValue::Null]
    );
    assert_eq!(
        a.shift(1, None).unwrap().data(),
        &vec![AttrValue::Null, AttrValue::Int(1)]
    );
}
#[test]
fn fractional_changes_preserve_small_integer_differences_and_missing_policy() {
    let a = column(vec![AttrValue::Int(i64::MAX - 1), AttrValue::Int(i64::MAX)]);
    let expected = 1. / (i64::MAX - 1) as f64;
    assert_eq!(
        a.pct_change(None).unwrap().data(),
        &floats(&[None, Some(expected)])
    );
    let a = column(vec![
        AttrValue::SmallInt(2),
        AttrValue::SmallInt(3),
        AttrValue::Null,
        AttrValue::SmallInt(0),
    ]);
    assert_eq!(
        a.pct_change(None).unwrap().data(),
        &floats(&[None, Some(0.5), None, None])
    );
    assert_eq!(
        a.pct_change(Some(0)).unwrap().data(),
        &floats(&[Some(0.), Some(0.), None, None])
    );
    assert_eq!(
        a.pct_change(Some(usize::MAX)).unwrap().data(),
        &vec![AttrValue::Null; 4]
    );
}
