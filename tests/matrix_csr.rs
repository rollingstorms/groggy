use groggy::storage::{matrix::unified_matrix::SparseMatrix, GraphMatrix};
#[test]
fn csr_reads_updates_clones_and_owned_dense_consumers_match_references() {
    let mut a =
        GraphMatrix::from_csr(2, 3, vec![2., 5., -1.], vec![0, 2, 1], vec![0, 2, 3]).unwrap();
    assert!(a.is_sparse());
    assert_eq!(a.zero_count().unwrap(), 3);
    assert_eq!(a.get_checked(0, 1).unwrap(), 0.);
    let saved = a.clone();
    let properties = groggy::storage::matrix::MatrixProperties::analyze(&saved);
    assert!(properties.is_sparse);
    assert_eq!(properties.sparsity, Some(0.5));
    assert_eq!(GraphMatrix::<f64>::zeros(2, 3).zero_count().unwrap(), 6);
    a.set(0, 1, 4.).unwrap();
    a.set(0, 0, 0.).unwrap();
    assert_eq!(a.get(0, 0), Some(0.));
    assert_eq!(a.get(0, 1), Some(4.));
    assert_eq!(saved.get(0, 0), Some(2.));
    assert_eq!(saved.get(0, 1), Some(0.));
    let b = GraphMatrix::from_row_major_data(vec![1., 2., 3.], 3, 1, None).unwrap();
    let product = saved.multiply(&b).unwrap();
    assert_eq!(product.get(0, 0), Some(17.));
    assert_eq!(product.get(1, 0), Some(-2.));
    assert!(!product.is_sparse());
    assert_eq!(saved.transpose().unwrap().get(2, 0), Some(5.));
    assert_eq!(saved.reshape(3, 2).unwrap().get(1, 0), Some(5.));
    assert_eq!(saved.select_columns(&[2, 0]).unwrap().get(0, 0), Some(5.));
    assert_eq!(
        GraphMatrix::<f64>::from_csr(0, 3, vec![], vec![], vec![0])
            .unwrap()
            .zero_count()
            .unwrap(),
        0
    );
}
#[test]
fn malformed_csr_is_rejected_without_partial_construction_or_index_panics() {
    for (values, cols, ptr) in [
        (vec![1.], vec![0], vec![1, 1, 1]),
        (vec![1.], vec![3], vec![0, 1, 1]),
        (vec![1., 2.], vec![1, 1], vec![0, 2, 2]),
        (vec![1., 2.], vec![2, 1], vec![0, 2, 2]),
        (vec![0.], vec![0], vec![0, 1, 1]),
        (vec![1.], vec![0], vec![0, 2, 1]),
    ] {
        assert!(GraphMatrix::from_csr(2, 3, values, cols, ptr).is_err());
    }
    let mut sparse = SparseMatrix::new(1, 2, vec![1.], vec![1], vec![0, 1]).unwrap();
    sparse.row_pointers[1] = 100;
    assert!(sparse.get(0, 1).is_err());
    assert!(sparse.set(0, 0, 1.).is_err());
}
