use groggy::storage::{
    advanced_matrix::unified_matrix::{MatrixStorage, SparseMatrix, UnifiedMatrix},
    GraphMatrix,
};
#[test]
fn csr_arithmetic_products_and_selections_stay_canonical_without_dense_materialization() {
    let a = SparseMatrix::new(2, 3, vec![2i64, 3, 4], vec![0, 2, 1], vec![0, 2, 3]).unwrap();
    let b = SparseMatrix::new(2, 3, vec![-2i64, 5], vec![0, 2], vec![0, 1, 2]).unwrap();
    let sum = a.add(&b).unwrap();
    sum.validate().unwrap();
    assert_eq!(sum.to_dense_values().unwrap(), vec![0, 0, 3, 0, 4, 5]);
    let selected = a.select(&[1, 0, 1], &[2, 1, 2]).unwrap();
    assert_eq!(
        selected.to_dense_values().unwrap(),
        vec![0, 4, 0, 3, 0, 3, 0, 4, 0]
    );
    let transposed = a.transpose().unwrap();
    assert_eq!(
        transposed.to_dense_values().unwrap(),
        vec![2, 0, 0, 4, 3, 0]
    );
    let product = a.matmul(&transposed).unwrap();
    assert_eq!(product.to_dense_values().unwrap(), vec![13, 0, 0, 16]);
    assert!(SparseMatrix::new(1, 1, vec![i64::MAX], vec![0], vec![0, 1])
        .unwrap()
        .add(&SparseMatrix::new(1, 1, vec![1], vec![0], vec![0, 1]).unwrap())
        .is_err());
    let a = GraphMatrix::from_csr(2, 2, vec![2., 3.], vec![0, 1], vec![0, 1, 2]).unwrap();
    assert!(a.multiply(&a).unwrap().is_sparse());
    assert!(a.add(&a).unwrap().is_sparse());
    assert!(a.transpose().unwrap().is_sparse());
    assert!(a.select_indices(&[1, 0], &[0, 1]).unwrap().is_sparse());
    assert!(!a.to_dense().unwrap().is_sparse());
    assert!(a.to_dense().unwrap().to_csr().unwrap().is_sparse());
}
#[test]
fn sparse_products_retain_accurate_cancellation_and_ieee_implicit_zero_rules() {
    let a = SparseMatrix::new(1, 3, vec![1e308, 1., -1e308], vec![0, 1, 2], vec![0, 3]).unwrap();
    let b = SparseMatrix::new(3, 1, vec![1.; 3], vec![0; 3], vec![0, 1, 2, 3]).unwrap();
    assert_eq!(a.matmul(&b).unwrap().get(0, 0).unwrap(), 1.);
    let a = SparseMatrix::new(1, 2, vec![f64::INFINITY], vec![0], vec![0, 1]).unwrap();
    let b = SparseMatrix::new(2, 2, vec![1.], vec![0], vec![0, 1, 1]).unwrap();
    let out = a.matmul(&b).unwrap();
    assert!(out.get(0, 0).unwrap().is_infinite());
    assert!(out.get(0, 1).unwrap().is_nan());
    assert!(a
        .hadamard(&SparseMatrix::new(1, 2, vec![], vec![], vec![0, 0]).unwrap())
        .unwrap()
        .get(0, 0)
        .unwrap()
        .is_nan());
    let dense = UnifiedMatrix::from_data(vec![1., 2., 3., 4.], 2, 2).unwrap();
    let sparse = UnifiedMatrix::from_csr(2, 2, vec![2., 3.], vec![0, 1], vec![0, 1, 2]).unwrap();
    assert_eq!(
        sparse.matmul(&dense).unwrap().to_vec().unwrap(),
        vec![2., 4., 9., 12.]
    );
    assert_eq!(
        dense.matmul(&sparse).unwrap().to_vec().unwrap(),
        vec![2., 6., 6., 12.]
    );
}
#[test]
fn large_diagonal_product_storage_grows_with_nonzeros() {
    let n = 10000;
    let a =
        UnifiedMatrix::from_csr(n, n, vec![2i64; n], (0..n).collect(), (0..=n).collect()).unwrap();
    let out = a.matmul(&a).unwrap();
    if let MatrixStorage::Sparse(s) = out.storage {
        assert_eq!(s.values.len(), n);
        assert!(s.values.iter().all(|v| *v == 4));
    } else {
        panic!("unexpected dense output");
    }
    assert_eq!(a.sum().unwrap(), 20000);
    assert_eq!(a.min().unwrap(), 0);
}

#[test]
fn scalar_maps_reshape_and_zero_normalization_follow_canonical_storage() {
    let a = GraphMatrix::from_csr(2, 3, vec![2., 3.], vec![0, 2], vec![0, 1, 2]).unwrap();
    let scaled = a.scalar_multiply(2.).unwrap();
    assert!(scaled.is_sparse());
    assert_eq!(scaled.get(1, 2), Some(6.));
    let shifted = a.scalar_add(1.).unwrap();
    assert!(shifted.is_sparse());
    assert_eq!(shifted.get(0, 1), Some(1.));
    let reshaped = a.reshape(3, 2).unwrap();
    assert!(reshaped.is_sparse());
    assert_eq!(reshaped.get(2, 1), Some(3.));
    assert!(!a.dense().unwrap().is_sparse());
    let tiny = f64::from_bits(1);
    let a = SparseMatrix::new(1, 1, vec![-tiny], vec![0], vec![0, 1]).unwrap();
    let b = SparseMatrix::new(1, 1, vec![0.5], vec![0], vec![0, 1]).unwrap();
    let out = a.matmul(&b).unwrap();
    assert!(out.values.is_empty());
    assert_eq!(out.get(0, 0).unwrap().to_bits(), 0);
}
