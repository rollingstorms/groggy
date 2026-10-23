use groggy::storage::array::{base::BaseArray as LegacyArray, BaseArray};
use groggy::{AttrValue as V, AttrValueType};

fn permutations(values: &[V]) -> Vec<Vec<V>> {
    if values.is_empty() {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    for i in 0..values.len() {
        let mut rest = values.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            result.push(tail);
        }
    }
    result
}
#[test]
fn sums_preserve_integer_cancellation_and_validate_final_output_range() {
    for (values, expected) in [
        (
            vec![V::Int(i64::MAX), V::SmallInt(1), V::Int(-i64::MAX)],
            V::Int(1),
        ),
        (
            vec![V::Int(i64::MAX), V::Int(i64::MAX), V::Int(-i64::MAX)],
            V::Int(i64::MAX),
        ),
        (
            vec![
                V::Int(i64::MAX),
                V::Float(-(2.0_f32).powi(63)),
                V::SmallInt(2),
            ],
            V::Float64(1.0),
        ),
        (
            vec![V::Float(1e20), V::Float(1.0), V::Float(-1e20)],
            V::Float64(1.0),
        ),
        (
            vec![V::Int(i64::MAX), V::SmallInt(1), V::Float(0.0)],
            V::Float64((2.0_f64).powi(63)),
        ),
    ] {
        for values in permutations(&values) {
            let array = BaseArray::new(values.clone());
            let legacy = LegacyArray::new(values, AttrValueType::Float);
            assert_eq!(array.sum().unwrap(), expected);
            assert_eq!(legacy.sum().unwrap(), expected);
            assert_eq!(array.mean().unwrap(), legacy.mean().unwrap());
        }
    }
    for values in [
        vec![V::Int(i64::MAX), V::Int(1)],
        vec![V::Int(i64::MIN), V::Int(-1)],
        vec![V::Float64(f64::MAX); 2],
    ] {
        assert!(BaseArray::new(values.clone()).sum().is_err());
        assert!(LegacyArray::new(values, AttrValueType::Int).sum().is_err());
    }
    let large = BaseArray::new(vec![V::Float(f32::MAX); 2]);
    assert_eq!(large.mean().unwrap(), f64::from(f32::MAX));
    assert_eq!(large.var().unwrap(), 0.0);
}
#[test]
fn high_offset_sample_variance_matches_exact_integer_reference() {
    for offset in [0, i64::MAX - 4, i64::MIN + 4] {
        for length in 2_u32..=5 {
            for pattern in 0..3_usize.pow(length) {
                let mut digits = pattern;
                let values: Vec<i64> = (0..length)
                    .map(|_| {
                        let value = [-3, 0, 2][digits % 3];
                        digits /= 3;
                        value
                    })
                    .collect();
                let count = i128::from(length);
                let sum: i128 = values.iter().map(|v| i128::from(*v)).sum();
                let squares: i128 = values.iter().map(|v| i128::from(*v).pow(2)).sum();
                let expected = (count * squares - sum * sum) as f64 / (count * (count - 1)) as f64;
                let array = BaseArray::new(values.iter().map(|v| V::Int(offset + v)).collect());
                assert!(
                    (array.var().unwrap() - expected).abs() < 1e-13,
                    "offset={offset}, values={values:?}"
                );
                assert!((array.std().unwrap() - expected.sqrt()).abs() < 1e-13);
            }
        }
    }
    for values in permutations(&[
        V::Int(i64::MAX),
        V::Float((2.0_f32).powi(63)),
        V::Int(i64::MAX - 1),
    ]) {
        assert_eq!(BaseArray::new(values).var().unwrap(), 1.0);
    }
}
#[test]
fn medians_keep_exact_integer_order_and_even_pair_cancellation() {
    for values in [
        vec![V::Int(i64::MIN), V::Int(i64::MAX)],
        vec![V::Int(i64::MAX), V::Float(-(2.0_f32).powi(63))],
    ] {
        assert_eq!(BaseArray::new(values).median().unwrap(), -0.5);
    }
    let array = BaseArray::new(vec![V::Float(f32::MAX), V::Float(-f32::MAX), V::Null]);
    assert_eq!(array.median().unwrap(), 0.0);
    assert_eq!(array.mean().unwrap(), 0.0);
    let expected = 2.0 * f64::from(f32::MAX).powi(2);
    assert_eq!(array.var().unwrap(), expected);
    assert!((array.std().unwrap() / expected.sqrt() - 1.).abs() < 1e-15);
    let values = vec![V::Int(5), V::Null, V::SmallInt(1), V::Float(2.0)];
    let array = BaseArray::new(values.clone());
    assert_eq!(array.median().unwrap(), 2.0);
    assert_eq!(array.inner.as_ref(), &values);
}
#[test]
fn null_empty_nonnumeric_and_nonfinite_domains_are_explicit() {
    for values in [
        vec![],
        vec![V::Null; 3],
        vec![V::Int(1), V::Bool(false)],
        vec![V::Text("no".into())],
        vec![V::Float(f32::NAN)],
        vec![V::Float(f32::INFINITY)],
        vec![V::Float(f32::NEG_INFINITY)],
    ] {
        let array = BaseArray::new(values.clone());
        assert!(array.sum().is_err());
        assert!(array.mean().is_err());
        assert!(array.median().is_err());
        assert!(array.var().is_err());
        assert!(array.std().is_err());
        let legacy = LegacyArray::new(values, AttrValueType::Float);
        assert!(legacy.sum().is_err());
        assert!(legacy.mean().is_err());
    }
    let one = BaseArray::new(vec![V::Null, V::SmallInt(7), V::Null]);
    assert_eq!(one.sum().unwrap(), V::Int(7));
    assert_eq!(one.mean().unwrap(), 7.0);
    assert_eq!(one.median().unwrap(), 7.0);
    assert!(one.var().is_err());
    assert!(one.std().is_err());
}

#[test]
fn extrema_retain_exact_selected_value_type_and_first_tie() {
    for values in permutations(&[V::Int(i64::MAX - 1), V::Int(i64::MAX), V::Null]) {
        let array = BaseArray::new(values.clone());
        assert_eq!(array.min().unwrap(), V::Int(i64::MAX - 1));
        assert_eq!(array.max().unwrap(), V::Int(i64::MAX));
        let legacy = LegacyArray::new(values, AttrValueType::Int);
        assert_eq!(legacy.min().unwrap(), array.min().unwrap());
        assert_eq!(legacy.max().unwrap(), array.max().unwrap());
    }
    for values in [
        vec![V::Int(i64::MAX), V::Float((2.0_f32).powi(63))],
        vec![V::Int(16_777_217), V::Float(16_777_216.0)],
    ] {
        for order in permutations(&values) {
            let expected_min = if matches!(values[0], V::Int(i64::MAX)) {
                values[0].clone()
            } else {
                values[1].clone()
            };
            let expected_max = if matches!(values[0], V::Int(i64::MAX)) {
                values[1].clone()
            } else {
                values[0].clone()
            };
            let array = BaseArray::new(order.clone());
            let legacy = LegacyArray::new(order, AttrValueType::Float);
            assert_eq!(array.min().unwrap(), expected_min);
            assert_eq!(legacy.min().unwrap(), expected_min);
            assert_eq!(array.max().unwrap(), expected_max);
            assert_eq!(legacy.max().unwrap(), expected_max);
        }
    }
    let tied = BaseArray::new(vec![V::Float(-0.0), V::SmallInt(0), V::Float(0.0)]);
    assert!(matches!(tied.min().unwrap(),V::Float(v) if v.is_sign_negative()));
    assert!(matches!(tied.max().unwrap(),V::Float(v) if v.is_sign_negative()));
}
#[test]
fn extrema_validate_every_element_and_keep_legacy_text_bool_domains() {
    for value in [
        V::Float(f32::NAN),
        V::Float(f32::INFINITY),
        V::Float(f32::NEG_INFINITY),
    ] {
        for values in [
            vec![value.clone()],
            vec![V::Int(0), value.clone()],
            vec![value, V::Int(0)],
        ] {
            let array = BaseArray::new(values.clone());
            let legacy = LegacyArray::new(values, AttrValueType::Float);
            assert!(array.min().is_err());
            assert!(array.max().is_err());
            assert!(legacy.min().is_err());
            assert!(legacy.max().is_err());
        }
    }
    for values in [
        vec![],
        vec![V::Null],
        vec![V::Bool(true)],
        vec![V::Text("a".into())],
    ] {
        let array = BaseArray::new(values);
        assert!(array.min().is_err());
        assert!(array.max().is_err());
    }
    let legacy = LegacyArray::new(
        vec![
            V::Text("z".into()),
            V::Null,
            V::CompactText(groggy::types::CompactString::new("a")),
        ],
        AttrValueType::Text,
    );
    assert_eq!(
        legacy.min().unwrap(),
        V::CompactText(groggy::types::CompactString::new("a"))
    );
    assert_eq!(legacy.max().unwrap(), V::Text("z".into()));
    let legacy = LegacyArray::new(vec![V::Bool(true), V::Bool(false)], AttrValueType::Bool);
    assert_eq!(legacy.min().unwrap(), V::Bool(false));
    assert_eq!(legacy.max().unwrap(), V::Bool(true));
    let mixed = LegacyArray::new(vec![V::Int(0), V::Text("x".into())], AttrValueType::Int);
    assert!(mixed.min().is_err());
    assert!(mixed.max().is_err());
}
