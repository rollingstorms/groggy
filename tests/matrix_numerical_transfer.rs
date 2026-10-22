use groggy::storage::{matrix::UnifiedMatrix, GraphMatrix, NumericalMatrixData};
fn assert_send<T: Send>() {}
#[test]
fn owned_numerical_data_transfer_preserves_values_storage_and_labels() {
    assert_send::<NumericalMatrixData<f64>>();
    let mut a = GraphMatrix::from_csr(2, 3, vec![2., 4.], vec![0, 2], vec![0, 1, 2]).unwrap();
    a.set_column_names_result(vec!["a".into(), "b".into(), "c".into()])
        .unwrap();
    let snapshot = a.numerical_data().unwrap();
    a.set(0, 0, 99.).unwrap();
    let snapshot = std::thread::spawn(move || {
        let a = snapshot.into_matrix().unwrap();
        assert!(a.is_sparse());
        assert_eq!(a.get(0, 0), Some(2.));
        a.numerical_data().unwrap()
    })
    .join()
    .unwrap()
    .into_matrix()
    .unwrap();
    assert_eq!(snapshot.column_names(), a.column_names());
    assert_eq!(snapshot.get(1, 2), Some(4.));
    assert!(!snapshot.requires_grad_enabled());
    let tracked = GraphMatrix::<f64>::ones(1, 1).requires_grad(true);
    assert!(!tracked
        .numerical_data()
        .unwrap()
        .into_matrix()
        .unwrap()
        .requires_grad_enabled());
}
#[test]
fn numerical_data_validates_logical_metadata() {
    let mut a = UnifiedMatrix::<f64>::new(2, 2).unwrap();
    a.strides.row_stride = usize::MAX;
    assert!(GraphMatrix::from_storage(a).numerical_data().is_err());
}
