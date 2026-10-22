use groggy::storage::{
    matrix::unified_matrix::SparseMatrix,
    numeric::scalar::{NumericComparison, NumericInput},
    GraphMatrix,
};
fn diagonal(n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_csr(n, n, vec![2.; n], (0..n).collect(), (0..=n).collect()).unwrap()
}
#[test]
fn sparse_zero_preserving_operations_do_not_trigger_dense_limits() {
    let a = diagonal(10000);
    for result in [
        a.sqrt().unwrap(),
        a.scalar_multiply(2.).unwrap(),
        a.comparison_mask(NumericInput::Float(0.), NumericComparison::Greater)
            .unwrap()
            .cast::<f64>()
            .unwrap(),
    ] {
        assert!(result.is_sparse());
        assert_eq!(result.shape(), (10000, 10000));
    }
}
#[test]
fn nonzero_image_and_nonfinite_scale_reject_before_large_grid_work() {
    let a = diagonal(10000);
    for error in [
        a.exp().unwrap_err(),
        a.log().unwrap_err(),
        a.scalar_add(1.).unwrap_err(),
        a.scalar_multiply(f64::INFINITY).unwrap_err(),
        a.comparison_mask(NumericInput::Float(0.), NumericComparison::GreaterEqual)
            .unwrap_err(),
    ] {
        assert!(error.to_string().contains("to_dense"), "{error}");
    }
    let lower = SparseMatrix::new(10000, 10000, vec![], vec![], vec![0; 10001]).unwrap();
    assert!(lower
        .scale(f64::INFINITY)
        .unwrap_err()
        .to_string()
        .contains("to_dense"));
    assert_eq!(a.get(0, 0), Some(2.));
}
#[test]
fn nonfinite_sparse_and_mixed_products_guard_large_operands() {
    let n = 10000;
    let a = diagonal(n);
    let mut values = vec![2.; n];
    values[0] = f64::INFINITY;
    let b = GraphMatrix::from_csr(n, n, values, (0..n).collect(), (0..=n).collect()).unwrap();
    assert!(a.multiply(&b).unwrap_err().to_string().contains("to_dense"));
    let row = GraphMatrix::from_row_major_data(vec![f64::INFINITY; n], 1, n, None).unwrap();
    assert!(row
        .multiply(&a)
        .unwrap_err()
        .to_string()
        .contains("to_dense"));
    assert!(a
        .multiply(&row.transpose().unwrap())
        .unwrap_err()
        .to_string()
        .contains("to_dense"));
}
#[test]
fn explicit_dense_opt_in_and_small_sparse_ieee_results_remain_available() {
    let n = 1025;
    let a = GraphMatrix::<f64>::from_csr(n, n, vec![], vec![], vec![0; n + 1]).unwrap();
    assert!(a.scalar_add(1.).is_err());
    let dense = a.to_dense().unwrap().scalar_add(1.).unwrap();
    assert_eq!(dense.get(1024, 1024), Some(1.));
    assert!(!dense.is_sparse());
    let a = diagonal(2);
    assert_eq!(a.exp().unwrap().get(0, 1), Some(1.));
    let scaled = a.scalar_multiply(f64::INFINITY).unwrap();
    assert!(scaled.get(0, 1).unwrap().is_nan());
    assert_eq!(scaled.get(0, 0), Some(f64::INFINITY));
}
