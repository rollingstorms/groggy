use groggy::storage::{
    advanced_matrix::{MatrixLayout, UnifiedMatrix},
    matrix::{MatrixIndex, MatrixSlice, MatrixSlicing},
    GraphMatrix,
};
fn labelled() -> GraphMatrix<f64> {
    let mut storage =
        UnifiedMatrix::<f64>::new_with_layout(2, 3, MatrixLayout::ColumnMajor).unwrap();
    for row in 0..2 {
        for col in 0..3 {
            storage.set(row, col, (row * 3 + col) as f64).unwrap();
        }
    }
    let mut matrix = GraphMatrix::from_storage(storage);
    matrix.set_row_labels(Some(vec!["r0".into(), "r1".into()]));
    matrix.set_column_names(vec!["a".into(), "b".into(), "c".into()]);
    matrix
}
#[test]
fn selections_are_owned_ordered_labelled_and_checked_including_empty_shapes() {
    let source = labelled();
    let mut selected = source.select_indices(&[1, 0, 1], &[2, 0]).unwrap();
    assert_eq!(selected.shape(), (3, 2));
    assert_eq!(selected.get(0, 0), Some(5.));
    assert_eq!(selected.get(1, 1), Some(0.));
    assert_eq!(selected.column_names(), &["c".to_string(), "a".to_string()]);
    assert_eq!(
        selected.row_labels().unwrap(),
        &["r1".to_string(), "r0".to_string(), "r1".to_string()]
    );
    selected.set(0, 0, 100.).unwrap();
    assert_eq!(source.get(1, 2), Some(5.));
    assert!(source.select_columns(&[3]).is_err());
    assert!(source.select_indices(&[2], &[]).is_err());
    assert_eq!(source.select_columns(&[]).unwrap().shape(), (2, 0));
    assert_eq!(source.select_indices(&[], &[1]).unwrap().shape(), (0, 1));
    assert!(MatrixSlicing::get_rows(&source, &[usize::MAX]).is_err());
    let reverse = source
        .get_submatrix(&MatrixSlice::new(
            MatrixIndex::Range {
                start: None,
                stop: None,
                step: Some(-1),
            },
            MatrixIndex::All,
        ))
        .unwrap();
    assert_eq!(reverse.get(0, 0), Some(3.));
}
#[test]
fn reshape_uses_logical_row_order_for_column_major_storage_and_transpose_swaps_identity() {
    let source = labelled();
    let reshaped = source.reshape(3, 2).unwrap();
    for row in 0..3 {
        for col in 0..2 {
            assert_eq!(reshaped.get(row, col), Some((row * 2 + col) as f64));
        }
    }
    assert!(reshaped.row_labels().is_none());
    assert_eq!(
        source.reshape(2, 3).unwrap().row_labels(),
        source.row_labels()
    );
    assert!(source.reshape(usize::MAX, 2).is_err());
    assert_eq!(
        GraphMatrix::<f64>::from_row_major_data(vec![], 3, 0, None)
            .unwrap()
            .reshape(0, 4)
            .unwrap()
            .shape(),
        (0, 4)
    );
    let transpose = source.transpose().unwrap();
    assert_eq!(transpose.shape(), (3, 2));
    assert_eq!(transpose.row_labels().unwrap(), source.column_names());
    assert_eq!(transpose.column_names(), source.row_labels().unwrap());
    assert_eq!(transpose.get(2, 1), Some(5.));
}
