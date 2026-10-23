use groggy::{
    storage::{array::BaseArray, table::BaseTable},
    AttrValue as V,
};
fn flags(values: Vec<V>, factor: Option<f64>) -> Result<Vec<V>, groggy::GraphError> {
    BaseTable::from_columns(std::collections::HashMap::from([(
        "x".into(),
        BaseArray::new(values),
    )]))?
    .check_outliers("x", factor)
    .map(|a| a.data().to_vec())
}
fn expected(values: &[bool]) -> Vec<V> {
    values.iter().map(|&v| V::Bool(v)).collect()
}
#[test]
fn outliers_preserve_f64_high_integer_and_mixed_width_distinctions() {
    let maximum = i64::MAX;
    assert_eq!(
        flags(
            [
                maximum - 100,
                maximum - 4,
                maximum - 3,
                maximum - 2,
                maximum - 1,
                maximum,
                maximum,
                maximum
            ]
            .map(V::Int)
            .to_vec(),
            None
        )
        .unwrap(),
        expected(&[true, false, false, false, false, false, false, false])
    );
    assert_eq!(
        flags(
            vec![
                V::Int(maximum - 2),
                V::Int(maximum - 1),
                V::Int(maximum),
                V::Float64(2f64.powi(63))
            ],
            Some(0.)
        )
        .unwrap(),
        expected(&[true, false, false, false])
    );
    let unit = 2f64.powi(-40);
    assert_eq!(
        flags(
            [0., 1., 2., 3., 4., 5., 6., 100.]
                .map(|v| V::Float64(1. + unit * v))
                .to_vec(),
            None
        )
        .unwrap(),
        expected(&[false, false, false, false, false, false, false, true])
    );
}
#[test]
fn outlier_fences_need_not_fit_in_f64_and_keep_strict_boundaries() {
    assert_eq!(
        flags(vec![V::Float64(f64::MAX); 4], Some(f64::MAX)).unwrap(),
        expected(&[false; 4])
    );
    assert_eq!(
        flags(
            vec![
                V::Float64(-f64::MAX),
                V::Int(-1),
                V::Int(0),
                V::Int(1),
                V::Float64(f64::MAX)
            ],
            None
        )
        .unwrap(),
        expected(&[true, false, false, false, true])
    );
    // q1=1, q3=3, fences=-2 and 6: equality is inlier.
    assert_eq!(
        flags([-2, 0, 1, 1, 2, 3, 3, 6].map(V::Int).to_vec(), None).unwrap(),
        expected(&[false; 8])
    );
    let tiny = f64::from_bits(1);
    assert_eq!(
        flags([-tiny, 0., 0., tiny].map(V::Float64).to_vec(), Some(tiny)).unwrap(),
        expected(&[true, false, false, false])
    );
}
#[test]
fn outlier_validation_and_missing_cells_are_explicit_and_row_aligned() {
    assert_eq!(
        flags(
            vec![
                V::Int(1),
                V::Null,
                V::Text("skip".into()),
                V::Bool(true),
                V::Float64Vec(vec![2.])
            ],
            None
        )
        .unwrap(),
        expected(&[false; 5])
    );
    for factor in [-1., f64::NAN, f64::INFINITY] {
        assert!(flags(vec![V::Int(1)], Some(factor)).is_err());
    }
    for value in [
        V::Float(f32::NAN),
        V::Float64(f64::NAN),
        V::Float64(f64::INFINITY),
    ] {
        assert!(flags(vec![value], None).is_err());
    }
    assert!(flags(vec![V::Null, V::Text("skip".into())], None).is_err());
    assert!(flags(vec![], None).is_err());
}

#[test]
fn adaptive_fences_match_independent_integer_rational_reference() {
    for count in 1..=32 {
        let input: Vec<i64> = (0..count)
            .map(|i| i64::MAX - ((i * 7919 + count * 13) % 10_000) as i64)
            .collect();
        let mut sorted = input.clone();
        sorted.sort_unstable();
        let q1 = i128::from(sorted[count / 4]);
        let q3 = i128::from(sorted[3 * count / 4]);
        for numerator in [0i128, 1, 2, 6, 32] {
            let lower = 4 * q1 - numerator * (q3 - q1);
            let upper = 4 * q3 + numerator * (q3 - q1);
            let reference: Vec<_> = input
                .iter()
                .map(|&v| {
                    let v = 4 * i128::from(v);
                    V::Bool(v < lower || v > upper)
                })
                .collect();
            assert_eq!(
                flags(
                    input.iter().copied().map(V::Int).collect(),
                    Some(numerator as f64 / 4.)
                )
                .unwrap(),
                reference
            );
        }
    }
}
