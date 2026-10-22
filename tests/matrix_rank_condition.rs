use groggy::storage::GraphMatrix;
#[test]
fn rank_scans_full_rectangular_spectrum_and_is_common_scale_invariant() {
    for scale in [1e-300, 1., 1e300] {
        let a = GraphMatrix::from_row_major_data(vec![0., scale, 0., 0., 0., scale], 2, 3, None)
            .unwrap();
        assert_eq!(a.rank().unwrap(), 2);
        assert!((a.condition_number().unwrap() - 1.).abs() < 1e-14);
    }
    let a = GraphMatrix::<f32>::from_row_major_data(vec![1., 0., 0., 1e-8], 2, 2, None).unwrap();
    assert_eq!(a.rank().unwrap(), 1);
    assert_eq!(a.rank_with_tolerance(1e-9).unwrap(), 2);
    let a = GraphMatrix::<f64>::from_row_major_data(vec![1., 0., 0., 1e-8], 2, 2, None).unwrap();
    assert_eq!(a.rank().unwrap(), 2);
    assert!((a.condition_number().unwrap() / 1e8 - 1.).abs() < 1e-14);
    assert_eq!(GraphMatrix::<f64>::zeros(0, 3).rank().unwrap(), 0);
    assert!(GraphMatrix::<f64>::zeros(0, 3).condition_number().is_err());
}
#[test]
fn conditioning_and_tolerance_domains_are_explicit() {
    let a = GraphMatrix::<f64>::zeros(3, 3);
    assert_eq!(a.rank().unwrap(), 0);
    assert_eq!(a.condition_number().unwrap(), f64::INFINITY);
    for tolerance in [-1., f64::NAN, f64::INFINITY] {
        assert!(a.rank_with_tolerance(tolerance).is_err());
    }
    for value in [f64::NAN, f64::INFINITY] {
        let a = GraphMatrix::from_row_major_data(vec![value], 1, 1, None).unwrap();
        assert!(a.rank().is_err());
        assert!(a.condition_number().is_err());
    }
    let a = GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None).unwrap();
    assert!(a.rank().is_err());
    let a = GraphMatrix::from_row_major_data(vec![f64::MAX; 2], 2, 1, None).unwrap();
    assert_eq!(a.rank().unwrap(), 1);
    assert_eq!(a.condition_number().unwrap(), 1.);
}
