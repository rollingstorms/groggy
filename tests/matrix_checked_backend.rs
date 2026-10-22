use groggy::storage::{
    matrix::{MatrixLayout, UnifiedMatrix},
    GraphMatrix,
};
#[test]
fn lower_level_arithmetic_uses_logical_layout_and_exact_checked_dots() {
    let mut a = UnifiedMatrix::<f64>::new_with_layout(2, 2, MatrixLayout::ColumnMajor).unwrap();
    for (r, c, v) in [(0, 0, 1.), (0, 1, 2.), (1, 0, 3.), (1, 1, 4.)] {
        a.set(r, c, v).unwrap();
    }
    assert_eq!(a.to_vec().unwrap(), vec![1., 2., 3., 4.]);
    let b = UnifiedMatrix::from_data(vec![4., 3., 2., 1.], 2, 2).unwrap();
    assert_eq!(a.add(&b).unwrap().to_vec().unwrap(), vec![5.; 4]);
    assert_eq!(
        a.matmul(&b).unwrap().to_vec().unwrap(),
        vec![8., 5., 20., 13.]
    );
    assert_eq!(
        a.transpose().unwrap().to_vec().unwrap(),
        vec![1., 3., 2., 4.]
    );
    let a = UnifiedMatrix::from_data(vec![1e308, 1., -1e308], 1, 3).unwrap();
    let b = UnifiedMatrix::from_data(vec![1.; 3], 3, 1).unwrap();
    assert_eq!(a.matmul(&b).unwrap().get(0, 0).unwrap(), 1.);
    let a = UnifiedMatrix::from_data(vec![i64::MAX, 1, -i64::MAX], 1, 3).unwrap();
    assert_eq!(a.sum().unwrap(), 1);
    assert!(UnifiedMatrix::from_data(vec![i64::MAX], 1, 1)
        .unwrap()
        .add(&UnifiedMatrix::from_data(vec![1], 1, 1).unwrap())
        .is_err());
}
#[test]
fn sparse_and_malformed_metadata_have_explicit_behavior_and_matmul_alias_agrees() {
    let a = UnifiedMatrix::from_csr(2, 2, vec![2., 3.], vec![0, 1], vec![0, 1, 2]).unwrap();
    assert_eq!(a.sum().unwrap(), 5.);
    assert_eq!(a.min().unwrap(), 0.);
    assert_eq!(
        a.matmul(&a).unwrap().to_vec().unwrap(),
        vec![4., 0., 0., 9.]
    );
    let mut malformed = UnifiedMatrix::<f64>::new(1, 1).unwrap();
    malformed.strides.row_stride = usize::MAX;
    malformed.strides.col_stride = usize::MAX;
    malformed.shape.rows = 2;
    assert!(malformed.get(1, 0).is_err());
    assert!(malformed.set(1, 0, 2.).is_err());
    assert!(malformed.to_vec().is_err());
    let a = GraphMatrix::from_row_major_data(vec![1e308, 1., -1e308], 1, 3, None).unwrap();
    let b = GraphMatrix::from_row_major_data(vec![1.; 3], 3, 1, None).unwrap();
    assert_eq!(a.matmul(&b).unwrap().get(0, 0), Some(1.));
}

#[test]
fn safe_raw_backend_wrappers_validate_extents_types_and_preserve_outputs_on_errors() {
    use groggy::storage::matrix::backend::{ComputeBackendExt, NativeBackend};
    let backend = NativeBackend;
    let mut c = [7.];
    assert!(backend
        .gemm(&[1.], (1, 2), &[1., 2.], (2, 1), &mut c, (1, 1), 1., 0.)
        .is_err());
    assert_eq!(c, [7.]);
    assert!(backend
        .elementwise_add(&[1i64, 2], &[1i64], &mut [0i64, 0])
        .is_err());
    assert!(backend.reduce_sum(&[1i128, 2]).is_err());
    backend
        .gemm(
            &[1e308, 1., -1e308],
            (1, 3),
            &[1.; 3],
            (3, 1),
            &mut c,
            (1, 1),
            1.,
            0.,
        )
        .unwrap();
    assert_eq!(c, [1.]);
    let mut result = [99i64];
    assert!(backend
        .gemm(
            &[i64::MAX],
            (1, 1),
            &[2i64],
            (1, 1),
            &mut result,
            (1, 1),
            1,
            0
        )
        .is_err());
    assert_eq!(result, [99]);
    assert_eq!(backend.reduce_sum(&[i64::MAX, 1, -i64::MAX]).unwrap(), 1);
    assert!(backend.reduce_max(&[1., f64::NAN]).unwrap().is_nan());
    let mut result = [99i64, 98];
    assert!(backend
        .elementwise_add(&[1, i64::MAX], &[1, 1], &mut result)
        .is_err());
    assert_eq!(result, [99, 98]);
}
