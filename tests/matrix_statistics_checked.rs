use groggy::storage::{
    matrix::{operations::MatrixStats, MatrixLayout, NumericType, UnifiedMatrix},
    Axis, GraphMatrix,
};
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 2e-14 * b.abs().max(1.), "{a} != {b}");
}
#[test]
fn stable_norms_and_stats_preserve_extreme_magnitudes_and_integer_offsets() {
    let a = GraphMatrix::from_row_major_data(vec![1e308, 1e308], 1, 2, None).unwrap();
    close(a.norm().unwrap() / 1e308, 2f64.sqrt());
    assert_eq!(a.mean().unwrap(), 1e308);
    assert_eq!(a.norm_l1().unwrap(), f64::INFINITY);
    assert_eq!(a.norm_inf().unwrap(), 1e308);
    let a = UnifiedMatrix::from_data(vec![i64::MAX - 2, i64::MAX], 1, 2).unwrap();
    close(MatrixStats::std(&a).unwrap(), 1.);
    let a = GraphMatrix::from_storage(a);
    assert_eq!(a.mean_axis(Axis::Rows).unwrap(), vec![i64::MAX - 1]);
    for values in [
        vec![i128::MAX, i128::MAX],
        vec![i128::MIN, i128::MIN],
        vec![i128::MAX, i128::MIN],
    ] {
        let expected = if values[0] == values[1] { values[0] } else { 0 };
        assert_eq!(
            GraphMatrix::from_row_major_data(values, 1, 2, None)
                .unwrap()
                .mean_axis(Axis::Rows)
                .unwrap(),
            vec![expected]
        );
    }
    let a = GraphMatrix::from_row_major_data(vec![5i64, -2], 1, 2, None).unwrap();
    assert_eq!(a.mean_axis(Axis::Rows).unwrap(), vec![1]);
    assert!(<i64 as NumericType>::from_f64(2f64.powi(63)).is_none());
    assert!(<i128 as NumericType>::from_f64(2f64.powi(127)).is_none());
}
#[test]
fn sparse_statistics_match_dense_with_zeros_and_do_not_materialize_large_shapes() {
    let dense =
        GraphMatrix::from_row_major_data(vec![0., 2., -3., 0., 0., 5.], 2, 3, None).unwrap();
    let sparse = dense.to_csr().unwrap();
    close(sparse.mean().unwrap(), dense.mean().unwrap());
    for axis in [Axis::Rows, Axis::Columns] {
        for (a, b) in [
            (
                sparse.mean_axis(axis).unwrap(),
                dense.mean_axis(axis).unwrap(),
            ),
            (
                sparse.std_axis(axis).unwrap(),
                dense.std_axis(axis).unwrap(),
            ),
            (
                sparse.var_axis(axis).unwrap(),
                dense.var_axis(axis).unwrap(),
            ),
        ] {
            for (x, y) in a.into_iter().zip(b) {
                close(x, y);
            }
        }
        assert_eq!(
            sparse.min_axis(axis).unwrap(),
            dense.min_axis(axis).unwrap()
        );
        assert_eq!(
            sparse.max_axis(axis).unwrap(),
            dense.max_axis(axis).unwrap()
        );
        assert_eq!(
            sparse.sum_axis(axis).unwrap(),
            dense.sum_axis(axis).unwrap()
        );
    }
    let n = 10000;
    let a = GraphMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    close(a.mean().unwrap(), 4. / n as f64);
    close(a.norm().unwrap(), 400.);
    for value in a.std_axis(Axis::Columns).unwrap() {
        close(value, 4. / (n as f64).sqrt());
    }
    for value in a.column_variances().unwrap() {
        close(value, 16. * (n - 1) as f64 / (n * n) as f64);
    }
    let a = GraphMatrix::from_csr(1, 4, vec![f64::MAX, -f64::MAX], vec![0, 1], vec![0, 2]).unwrap();
    close(
        a.std_axis(Axis::Rows).unwrap()[0] / f64::MAX,
        (2f64 / 3.).sqrt(),
    );
}
#[test]
fn logical_layout_invalid_metadata_empty_and_nonfinite_have_explicit_behavior() {
    let mut a = UnifiedMatrix::<f64>::new_with_layout(2, 2, MatrixLayout::ColumnMajor).unwrap();
    for (r, c, v) in [(0, 0, 1.), (0, 1, 2.), (1, 0, 3.), (1, 1, 4.)] {
        a.set(r, c, v).unwrap();
    }
    assert_eq!(MatrixStats::row_means(&a).unwrap(), vec![1.5, 3.5]);
    assert_eq!(MatrixStats::col_means(&a).unwrap(), vec![2., 3.]);
    close(MatrixStats::std(&a).unwrap(), 1.25f64.sqrt());
    a.strides.row_stride = usize::MAX;
    assert!(MatrixStats::std(&a).is_err());
    assert!(GraphMatrix::from_storage(a).norm().is_err());
    let a = GraphMatrix::<f64>::from_csr(2, 0, vec![], vec![], vec![0, 0, 0]).unwrap();
    assert_eq!(a.norm().unwrap(), 0.);
    assert_eq!(a.mean_axis(Axis::Rows).unwrap(), vec![0., 0.]);
    let empty = UnifiedMatrix::<f64>::new(0, 0).unwrap();
    assert!(MatrixStats::mean(&empty).unwrap().is_nan());
    for a in [
        GraphMatrix::from_row_major_data(vec![f64::INFINITY, f64::NAN], 1, 2, None).unwrap(),
        GraphMatrix::from_csr(1, 2, vec![f64::INFINITY, f64::NAN], vec![0, 1], vec![0, 2]).unwrap(),
    ] {
        assert!(a.norm().unwrap().is_nan());
        assert!(a.norm_inf().unwrap().is_nan());
        assert!(a.norm_l1().unwrap().is_nan());
        assert!(a.mean().unwrap().is_nan());
        assert!(a.std_axis(Axis::Rows).unwrap()[0].is_nan());
    }
}
