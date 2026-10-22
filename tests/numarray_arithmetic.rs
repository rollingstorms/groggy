use groggy::storage::{
    array::{
        numeric_arithmetic::{compare_scalar, ConversionPolicy, NumericComparison, NumericInput},
        NumArray,
    },
    GraphMatrix,
};
#[test]
fn typed_vectors_keep_integer_precision_check_errors_and_offer_native_statistics() {
    let a = NumArray::new(vec![i64::MAX, 1, -i64::MAX]);
    let ones = NumArray::new(vec![1i64; 3]);
    assert_eq!(a.dot(&ones).unwrap(), 1);
    assert_eq!(a.sum(), 1.);
    assert!(a.checked_add(&ones).is_err());
    assert!(a.add(&ones).is_none());
    let a = NumArray::new(vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    assert_eq!(a.std_dev(), Some(1.));
    let a = NumArray::new(vec![1e308, 1., -1e308]);
    assert_eq!(a.dot(&NumArray::new(vec![1.; 3])).unwrap(), 1.);
    let a = NumArray::new(vec![3i32, 5]);
    assert_eq!(
        a.true_divide(&NumArray::new(vec![2i32, 2]))
            .unwrap()
            .as_slice(),
        &[1.5, 2.5]
    );
    assert!(a.true_divide(&NumArray::new(vec![0i32, 1])).is_err());
    assert_eq!(
        NumArray::<f64>::new(vec![])
            .dot(&NumArray::new(vec![]))
            .unwrap(),
        0.
    );
    assert_eq!(
        NumArray::new(vec![true, false])
            .checked_add(&NumArray::new(vec![false, true]))
            .unwrap()
            .as_slice(),
        &[true, true]
    );
}
#[test]
fn checked_conversions_comparisons_unique_and_matrix_casts_are_explicit() {
    let a = NumArray::new(vec![i64::MAX]);
    assert!(a.cast::<f64>().is_err());
    assert!(a.cast_with_policy::<f64>(ConversionPolicy::Round).is_ok());
    assert!(NumArray::new(vec![2147483648i64]).cast::<i32>().is_err());
    assert!(NumArray::new(vec![1e300])
        .cast_with_policy::<f32>(ConversionPolicy::Round)
        .is_err());
    assert_eq!(
        NumArray::new(vec![2.5, -2.5])
            .cast_with_policy::<i32>(ConversionPolicy::Round)
            .unwrap()
            .as_slice(),
        &[3, -3]
    );
    let comparisons = compare_scalar(
        &[NumericInput::Integer(i128::from(i64::MAX))],
        NumericInput::Float(i64::MAX as f64),
        NumericComparison::Less,
    );
    assert_eq!(comparisons, vec![true]);
    let a = NumArray::new(vec![i64::MAX, i64::MAX - 1, i64::MAX]);
    assert_eq!(a.unique().as_slice(), &[i64::MAX, i64::MAX - 1]);
    let a = GraphMatrix::from_csr(2, 2, vec![2i64, 3], vec![0, 1], vec![0, 1, 2]).unwrap();
    let cast = a.cast::<f64>().unwrap();
    assert!(cast.is_sparse());
    assert_eq!(cast.get(1, 1), Some(3.));
    assert!(GraphMatrix::from_row_major_data(vec![i64::MAX], 1, 1, None)
        .unwrap()
        .cast::<f64>()
        .is_err());
    assert!(NumArray::new(vec![1e308, 1e308]).norm().is_finite());
}
