use groggy::storage::{
    matrix::{
        ActivationFunctions, MathFunctions, MatrixLayout, MatrixOperations, MatrixStats,
        MatrixUtils, UnifiedMatrix,
    },
    GraphMatrix,
};
#[test]
fn math_functions_propagate_ieee_domains_and_reject_integer_loss_and_overflow() {
    let a = UnifiedMatrix::from_data(vec![-1., 0., 1., f64::NAN], 1, 4).unwrap();
    let b = MathFunctions::ln(&a).unwrap().to_vec().unwrap();
    assert!(b[0].is_nan() && b[3].is_nan());
    assert_eq!(b[1], f64::NEG_INFINITY);
    assert_eq!(b[2], 0.);
    assert!(MathFunctions::sqrt(&a).unwrap().get(0, 0).unwrap().is_nan());
    assert!(ActivationFunctions::relu(&a)
        .unwrap()
        .get(0, 3)
        .unwrap()
        .is_nan());
    assert!(MathFunctions::abs(&UnifiedMatrix::from_data(vec![i64::MIN], 1, 1).unwrap()).is_err());
    assert_eq!(
        MathFunctions::abs(&UnifiedMatrix::from_data(vec![i64::MIN + 1], 1, 1).unwrap())
            .unwrap()
            .get(0, 0)
            .unwrap(),
        i64::MAX
    );
    assert!(MathFunctions::ln(&UnifiedMatrix::from_data(vec![0i64], 1, 1).unwrap()).is_err());
    assert!(MathFunctions::sqrt(&UnifiedMatrix::from_data(vec![-1i64], 1, 1).unwrap()).is_err());
    // No implicit zero exists: ln(0) must not be evaluated for a full integer CSR.
    let a = UnifiedMatrix::from_csr(1, 1, vec![1i64], vec![0], vec![0, 1]).unwrap();
    assert_eq!(MathFunctions::ln(&a).unwrap().get(0, 0).unwrap(), 0);
}
#[test]
fn sparse_helpers_preserve_storage_and_checked_shapes() {
    let n = 10000;
    let a =
        UnifiedMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    let b = MathFunctions::sqrt(&a).unwrap();
    assert_eq!(MatrixStats::frobenius_norm(&b).unwrap().round(), 200.);
    let b = b.reshape(n * 2, n / 2).unwrap();
    assert!(matches!(
        b.storage,
        groggy::storage::matrix::unified_matrix::MatrixStorage::Sparse(_)
    ));
    assert_eq!(
        a.slice(1, 3, 1, 3).unwrap().to_vec().unwrap(),
        vec![4., 0., 0., 4.]
    );
    assert_eq!(a.slice(0, 0, 0, n).unwrap().shape().as_tuple(), (0, n));
    assert!(a.reshape(usize::MAX, 2).is_err());
    assert!(MatrixUtils::constant(usize::MAX, 2, 0f64).is_err());
    assert!(MatrixUtils::random::<f64>(usize::MAX, 2).is_err());
    assert!(MatrixUtils::from_fn(usize::MAX, 2, |_, _| 0f64).is_err());
    let a = GraphMatrix::from_storage(a);
    let b = a.concatenate(&a, 0).unwrap();
    assert!(b.is_sparse());
    assert_eq!(b.shape(), (n * 2, n));
    let chunks = b.split(&[n], 0).unwrap();
    assert!(chunks.iter().all(GraphMatrix::is_sparse));
    assert_eq!(chunks[1].get(n - 1, n - 1), Some(4.));
    let b = a.concatenate(&a, 1).unwrap();
    assert_eq!(b.shape(), (n, n * 2));
    assert!(b.is_sparse());
    assert_eq!(b.get(n - 1, n * 2 - 1), Some(4.));
}
#[test]
fn in_place_maps_use_logical_cells_and_validate_before_publication() {
    let mut a = UnifiedMatrix::<i64>::new_with_layout(2, 2, MatrixLayout::ColumnMajor).unwrap();
    for (r, c, v) in [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)] {
        a.set(r, c, v).unwrap();
    }
    a.map_elementwise_inplace(|v| *v += 1).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![2, 3, 4, 5]);
    a.strides.row_stride = usize::MAX;
    assert!(a.map_elementwise_inplace(|v| *v += 1).is_err());
    let a = GraphMatrix::from_storage(a);
    assert!(a.materialize().is_err());
    let a = GraphMatrix::from_row_major_data(vec![1., 2., 3., 4.], 2, 2, None).unwrap();
    let b = a.concatenate(&a, 1).unwrap();
    assert_eq!(b.shape(), (2, 4));
    assert_eq!(b.get(1, 3), Some(4.));
    assert!(a.split(&[0], 0).is_err());
    assert!(a.split(&[2, 1], 0).is_err());
}
