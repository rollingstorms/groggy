use groggy::storage::{
    array::{AdvancedIndexing, SliceIndex},
    matrix::{MatrixIndex, MatrixSlice, MatrixSlicing},
    GraphMatrix, NumArray,
};
fn range(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> SliceIndex {
    SliceIndex::Range { start, stop, step }
}
#[test]
fn slice_bounds_resolve_once_against_actual_length_and_extreme_steps_do_not_overflow() {
    for (index, expected) in [
        (range(Some(-2), None, None), vec![3, 4]),
        (range(None, Some(-2), None), vec![0, 1, 2]),
        (range(None, None, Some(-1)), vec![4, 3, 2, 1, 0]),
        (range(None, Some(-1), Some(-1)), vec![]),
        (range(Some(1), None, Some(i64::MAX)), vec![1]),
        (range(Some(4), None, Some(i64::MIN)), vec![4]),
        (
            range(Some(i64::MIN), Some(i64::MAX), None),
            vec![0, 1, 2, 3, 4],
        ),
    ] {
        assert_eq!(index.resolve_indices(5).unwrap(), expected);
    }
    assert!(range(None, None, Some(0)).validate().is_err());
    assert!(range(None, None, Some(0)).resolve_indices(5).is_err());
    assert!(SliceIndex::Single(i64::MIN).resolve_indices(5).is_err());
    assert_eq!(SliceIndex::Single(-1).resolve_indices(5).unwrap(), vec![4]);
    assert!(range(None, None, Some(-1))
        .resolve_indices(0)
        .unwrap()
        .is_empty());
    assert!(range(None, None, None).resolve_indices(usize::MAX).is_err());
    if usize::BITS == 64 {
        assert_eq!(
            range(None, None, Some(i64::MAX))
                .resolve_indices(usize::MAX)
                .unwrap(),
            vec![0, i64::MAX as usize, usize::MAX - 1]
        );
        assert_eq!(
            SliceIndex::Single(-1).resolve_indices(usize::MAX).unwrap(),
            vec![usize::MAX - 1]
        );
    }
}
#[test]
fn arrays_and_native_matrix_ranges_share_ordered_owned_selection() {
    let a = NumArray::new(vec![10i64, 20, 30, 40, 50]);
    let out = a.get_slice(&range(Some(-3), None, Some(-1))).unwrap();
    assert_eq!(out.as_slice(), &[30, 20, 10]);
    assert_eq!(a.as_slice(), &[10, 20, 30, 40, 50]);
    let a = GraphMatrix::from_row_major_data((0..12).map(f64::from).collect(), 4, 3, None).unwrap();
    let selection = MatrixSlice::new(
        MatrixIndex::Range {
            start: Some(-2),
            stop: None,
            step: None,
        },
        MatrixIndex::Range {
            start: None,
            stop: None,
            step: Some(-1),
        },
    );
    let out = a.get_submatrix(&selection).unwrap();
    assert_eq!(out.shape(), (2, 3));
    assert_eq!(out.get(0, 0), Some(8.));
    assert_eq!(out.get(1, 2), Some(9.));
    assert_eq!(
        MatrixSlicing::get_row(&a, 2).unwrap().as_slice(),
        &[6., 7., 8.]
    );
}
