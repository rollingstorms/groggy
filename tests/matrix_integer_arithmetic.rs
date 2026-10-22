use groggy::storage::{advanced_matrix::NumericType, GraphMatrix};
fn matrix<T: NumericType>(values: Vec<T>, rows: usize, cols: usize) -> GraphMatrix<T> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
#[test]
fn integer_elementwise_overflow_returns_errors_without_modifying_inputs() {
    let a = matrix(vec![i64::MAX, i64::MIN], 1, 2);
    let plus = matrix(vec![1, 0], 1, 2);
    let minus = matrix(vec![0, 1], 1, 2);
    assert!(a.add(&plus).is_err());
    assert!(a.subtract(&minus).is_err());
    assert!(a.elementwise_multiply(&matrix(vec![2, 1], 1, 2)).is_err());
    assert!((&a + &plus).is_err());
    assert_eq!(a.get(0, 0), Some(i64::MAX));
    assert!(matrix(vec![i128::MIN], 1, 1)
        .subtract(&matrix(vec![1], 1, 1))
        .is_err());
    let a = matrix(vec![1_i32, -3, 7], 1, 3);
    assert_eq!(
        a.add(&matrix(vec![2, 4, -2], 1, 3)).unwrap().get(0, 2),
        Some(5)
    );
    assert!(matrix(vec![i32::MAX], 1, 1)
        .elementwise_multiply(&matrix(vec![2], 1, 1))
        .is_err());
}
#[test]
fn integer_dots_preserve_cancellation_and_check_accumulator_and_output_bounds() {
    let a = matrix(vec![i64::MAX, i64::MAX, 1], 1, 3);
    let b = matrix(vec![2, -2, 1], 3, 1);
    assert_eq!((&a * &b).unwrap().get(0, 0), Some(1));
    assert!(matrix(vec![i64::MAX, 1], 1, 2)
        .multiply(&matrix(vec![1, 1], 2, 1))
        .is_err());
    assert!(matrix(vec![i128::MAX], 1, 1)
        .multiply(&matrix(vec![2], 1, 1))
        .is_err());
    assert_eq!(
        matrix(vec![i32::MAX, i32::MAX, 2], 1, 3)
            .multiply(&matrix(vec![2, -2, 1], 3, 1))
            .unwrap()
            .get(0, 0),
        Some(2)
    );
    assert_eq!(
        matrix(vec![true, false], 1, 2)
            .multiply(&matrix(vec![false, true], 2, 1))
            .unwrap()
            .get(0, 0),
        Some(false)
    );
    assert_eq!(
        matrix(Vec::<i64>::new(), 2, 0)
            .multiply(&matrix(vec![], 0, 1))
            .unwrap()
            .get(1, 0),
        Some(0)
    );
}
