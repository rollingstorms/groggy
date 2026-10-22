use groggy::storage::{
    matrix::UnifiedMatrix,
    array::numeric_arithmetic::{NumericComparison, NumericInput},
    GraphMatrix,
};
#[test]
fn public_typed_math_and_scalar_paths_are_checked_and_preserve_labels() {
    let mut a = GraphMatrix::from_row_major_data(vec![4i64, 9], 1, 2, None).unwrap();
    a.set_column_names_result(vec!["a".into(), "b".into()])
        .unwrap();
    assert_eq!(a.sqrt().unwrap().get(0, 1), Some(3));
    assert_eq!(a.sqrt().unwrap().column_names(), a.column_names());
    assert_eq!(a.scalar_add(1.).unwrap().get(0, 0), Some(5));
    assert_eq!(a.scalar_subtract(1.).unwrap().get(0, 1), Some(8));
    assert_eq!(a.scalar_multiply(2.).unwrap().get(0, 1), Some(18));
    assert!(a.scalar_multiply(0.5).is_err());
    assert!(a.scalar_divide(2.).is_err());
    let a = GraphMatrix::from_row_major_data(vec![i64::MIN], 1, 1, None).unwrap();
    assert!(a.abs().is_err());
    assert!(a.scalar_multiply(-1.).is_err());
    let a = GraphMatrix::from_row_major_data(vec![i64::MAX], 1, 1, None).unwrap();
    assert!(a.scalar_add(1.).is_err());
    let mask = a
        .comparison_mask(
            NumericInput::Integer(i128::from(i64::MAX)),
            NumericComparison::Equal,
        )
        .unwrap();
    assert_eq!(mask.get(0, 0), Some(true));
    let a = GraphMatrix::from_row_major_data(vec![0f32, 4.], 1, 2, None).unwrap();
    assert_eq!(a.log().unwrap().get(0, 0), Some(f32::NEG_INFINITY));
    assert_eq!(a.scalar_divide(2.).unwrap().get(0, 1), Some(2.));
}
#[test]
fn sparse_maps_comparisons_repetition_and_casts_preserve_storage_and_empty_shapes() {
    let n = 10000;
    let a = GraphMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    assert!(a.abs().unwrap().is_sparse());
    assert!(a.sqrt().unwrap().is_sparse());
    assert!(a.scalar_multiply(2.).unwrap().is_sparse());
    let mask = a
        .comparison_mask(NumericInput::Float(0.), NumericComparison::Greater)
        .unwrap();
    assert!(mask.is_sparse());
    assert_eq!(mask.get(0, 1), Some(false));
    assert!(mask.cast::<f64>().unwrap().is_sparse());
    let tiled = a.tile((1, 2)).unwrap();
    assert!(tiled.is_sparse());
    assert_eq!(tiled.shape(), (n, n * 2));
    assert_eq!(tiled.get(n - 1, n * 2 - 1), Some(4.));
    let repeated = a.repeat(2, 0).unwrap();
    assert!(repeated.is_sparse());
    assert_eq!(repeated.shape(), (n * 2, n));
    assert_eq!(repeated.get(n * 2 - 1, n - 1), Some(4.));
    let a = GraphMatrix::<f64>::zeros(3, 0);
    assert_eq!(a.tile((2, 4)).unwrap().shape(), (6, 0));
    assert_eq!(a.repeat(2, 1).unwrap().shape(), (3, 0));
    let a = GraphMatrix::<f64>::zeros(0, 3);
    assert_eq!(a.tile((2, 4)).unwrap().shape(), (0, 12));
    assert_eq!(a.repeat(0, 1).unwrap().shape(), (0, 0));
    let a = GraphMatrix::<f64>::ones(2, 2);
    assert!(a.tile((usize::MAX, 1)).is_err());
    assert!(a.repeat(usize::MAX, 1).is_err());
    assert!(a.repeat(1, 2).is_err());
}
#[test]
fn invalid_layouts_and_nonfinite_domains_do_not_substitute_zero_data() {
    let mut storage = UnifiedMatrix::<f64>::new(2, 2).unwrap();
    storage.strides.row_stride = usize::MAX;
    let a = GraphMatrix::from_storage(storage);
    assert!(a.abs().is_err());
    assert!(a.scalar_add(1.).is_err());
    assert!(a.tile((1, 1)).is_err());
    let a = GraphMatrix::from_row_major_data(vec![-1., 0., f64::NAN], 1, 3, None).unwrap();
    assert!(a.sqrt().unwrap().get(0, 0).unwrap().is_nan());
    assert!(a.log().unwrap().get(0, 0).unwrap().is_nan());
    assert_eq!(a.log().unwrap().get(0, 1), Some(f64::NEG_INFINITY));
    assert!(
        a.comparison_mask(NumericInput::Float(0.), NumericComparison::GreaterEqual)
            .unwrap()
            .get(0, 2)
            == Some(false)
    );
}
