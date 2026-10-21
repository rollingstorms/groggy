use groggy::storage::array::{numeric_reductions as k, simd_optimizations as s, NumArray};
fn equivalent(a: Option<f64>, b: Option<f64>) {
    match (a, b) {
        (Some(a), Some(b)) if a.is_nan() && b.is_nan() => {}
        _ => assert_eq!(a, b),
    }
}
#[test]
fn typed_kernels_retain_integer_subtotals_and_high_offset_differences() {
    assert_eq!(k::sum(&[i64::MAX, 1, -i64::MAX]), 1.0);
    assert_eq!(k::mean(&[i64::MAX, 1, -i64::MAX]), Some(1.0 / 3.0));
    for base in [i64::MAX - 2, i64::MIN] {
        let values = [base, base + 1, base + 2];
        assert_eq!(k::variance(&values), Some(1.0));
        assert_eq!(k::std_dev(&values), Some(1.0));
    }
    assert_eq!(k::median(&[i64::MIN, i64::MAX]), Some(-0.5));
    assert_eq!(k::sum(&[true, false, true]), 2.0);
    assert_eq!(k::variance(&[true, false]), Some(0.5));
    assert_eq!(k::sum(&[1e20_f32, 1.0, -1e20]), 1.0);
}
#[test]
fn scalar_simd_auto_and_direct_helpers_agree_across_dispatch_boundaries() {
    for n in [0, 1, 2, 3, 4, 7, 8, 9, 99, 100, 101, 256] {
        let data: Vec<_> = (0..n).map(|i| 1e16 + 2.0 * i as f64).collect();
        let array = NumArray::new(data.clone());
        assert_eq!(array.sum(), array.sum_simd());
        assert_eq!(array.sum(), array.sum_auto());
        equivalent(array.mean(), array.mean_simd());
        equivalent(array.mean(), array.mean_auto());
        equivalent(array.variance(), array.variance_simd());
        equivalent(array.std_dev(), array.std_dev_simd());
        equivalent(array.min(), array.min_simd());
        equivalent(array.max(), array.max_simd());
        equivalent(array.median(), array.median_optimized());
        equivalent(array.median(), array.median_auto());
        equivalent(array.mean(), s::simd_mean(&data));
        equivalent(array.std_dev(), s::simd_std_dev(&data));
        if n > 1 {
            let expected = n as f64 * (n as f64 + 1.0) / 3.0;
            assert!((array.variance().unwrap() - expected).abs() < 1e-12 * expected);
        }
    }
    assert_eq!(s::simd_variance(&[1.0, 3.0], 0.0), 10.0); // Explicit-center semantics remain.
}
#[test]
fn finite_range_rescaling_and_safe_medians_handle_extreme_values() {
    assert_eq!(k::sum(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX);
    assert_eq!(k::mean(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    assert_eq!(k::median(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    assert_eq!(k::median(&[-f64::MAX, f64::MAX]), Some(0.0));
    for scale in [1e-300, f64::MAX] {
        let data = [0.0, scale];
        let expected = scale / 2.0_f64.sqrt();
        assert!((k::std_dev(&data).unwrap() / expected - 1.0).abs() < 1e-15);
    }
    assert_eq!(k::variance(&[0.0, f64::MAX]), Some(f64::INFINITY));
    assert_eq!(k::variance(&[0.0, 1e-300]), Some(0.0));
    assert_eq!(k::variance(&[f64::MAX, f64::MAX]), Some(0.0));
}
#[test]
fn empty_singleton_and_nonfinite_contract_is_consistent_without_panics() {
    let empty: Vec<f64> = vec![];
    let array = NumArray::new(empty.clone());
    assert_eq!(array.sum(), 0.0);
    assert_eq!(array.mean(), None);
    assert_eq!(array.variance(), None);
    assert_eq!(array.std_dev(), None);
    assert_eq!(k::median(&empty), None);
    assert_eq!(k::variance(&[3.0]), Some(0.0));
    for data in [
        vec![f64::NAN],
        vec![0.0, f64::NAN],
        vec![f64::INFINITY],
        vec![f64::INFINITY, f64::NEG_INFINITY],
    ] {
        let array = NumArray::new(data.clone());
        assert!(array.variance().unwrap().is_nan());
        assert!(array.std_dev().unwrap().is_nan());
        equivalent(array.sum().into(), array.sum_simd().into());
        equivalent(array.mean(), array.mean_auto());
        equivalent(array.median(), array.median_auto());
        equivalent(array.min(), array.min_simd());
        if data.iter().any(|x| x.is_nan()) {
            assert!(array.min().unwrap().is_nan());
            assert!(array.max().unwrap().is_nan());
            assert!(array.median().unwrap().is_nan());
        }
    }
    assert_eq!(
        k::median(&[f64::NEG_INFINITY, -1.0]),
        Some(f64::NEG_INFINITY)
    );
    assert_eq!(k::median(&[1.0, f64::INFINITY]), Some(f64::INFINITY));
    let mut duplicates = vec![7.0; 20_000];
    assert_eq!(s::quickselect_median(&mut duplicates), Some(7.0));
}
