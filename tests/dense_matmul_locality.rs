//! Large dense dots preserve the shared numerical contract across logical layouts.
use groggy::storage::{
    matrix::{MatrixLayout, UnifiedMatrix},
    GraphMatrix, NumArray,
};

#[test]
fn contiguous_rhs_columns_match_checked_vector_dots_for_layouts_and_cancellation_values() {
    for layout in [MatrixLayout::RowMajor, MatrixLayout::ColumnMajor] {
        let mut left = UnifiedMatrix::<f64>::new_with_layout(3, 64, layout).unwrap();
        let mut right = UnifiedMatrix::<f64>::new_with_layout(64, 40, layout).unwrap();
        for i in 0..3 {
            for k in 0..64 {
                left.set(
                    i,
                    k,
                    match k % 4 {
                        0 => 1e16,
                        1 => 1.0 + i as f64,
                        2 => -1e16,
                        _ => 0.0,
                    },
                )
                .unwrap();
            }
        }
        for k in 0..64 {
            for j in 0..40 {
                right.set(k, j, (j % 7) as f64 - 3.0).unwrap();
            }
        }
        let left = GraphMatrix::try_from_storage(left).unwrap();
        let right = GraphMatrix::try_from_storage(right).unwrap();
        let actual = left.multiply(&right).unwrap();
        for i in 0..3 {
            for j in 0..40 {
                let row = NumArray::new((0..64).map(|k| left.get(i, k).unwrap()).collect());
                let column = NumArray::new((0..64).map(|k| right.get(k, j).unwrap()).collect());
                assert_eq!(
                    actual.get(i, j).unwrap().to_bits(),
                    row.dot(&column).unwrap().to_bits()
                );
            }
        }
    }
}
#[test]
fn dense_reordered_storage_preserves_checked_integer_overflow_and_nonfinite_zero_products() {
    let large = GraphMatrix::from_row_major_data(vec![i64::MAX; 64], 1, 64, None).unwrap();
    let rhs = GraphMatrix::from_row_major_data(vec![1i64; 64 * 40], 64, 40, None).unwrap();
    assert!(large.multiply(&rhs).is_err());
    let zeros = GraphMatrix::<f64>::zeros(1, 64);
    let rhs = GraphMatrix::from_row_major_data(vec![f64::INFINITY; 64 * 40], 64, 40, None).unwrap();
    let result = zeros.multiply(&rhs).unwrap();
    for col in 0..40 {
        assert!(result.get(0, col).unwrap().is_nan());
    }
}
