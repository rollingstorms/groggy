use groggy::storage::{
    array::{numeric_arithmetic::ConversionPolicy, NumArray},
    GraphMatrix,
};
use groggy::types::AttrValue;
#[test]
fn checked_constructors_never_substitute_a_different_shape() {
    for result in [
        GraphMatrix::<f64>::try_zeros(usize::MAX, 2),
        GraphMatrix::try_ones(usize::MAX, 2),
        GraphMatrix::try_eye(usize::MAX),
        GraphMatrix::try_zeros(0, usize::MAX),
        GraphMatrix::from_row_major_data(vec![], 0, usize::MAX, None),
        GraphMatrix::from_csr(0, usize::MAX, vec![], vec![], vec![0]),
    ] {
        assert!(result.is_err());
    }
    for (rows, cols) in [(0, 0), (0, 3), (3, 0), (2, 3)] {
        assert_eq!(
            GraphMatrix::<f64>::try_zeros(rows, cols).unwrap().shape(),
            (rows, cols)
        );
        assert_eq!(
            GraphMatrix::<f64>::try_ones(rows, cols).unwrap().shape(),
            (rows, cols)
        );
    }
}
#[test]
fn strict_attribute_columns_preserve_compact_ints_and_reject_precision_loss() {
    let a = NumArray::<f64>::from_attributes(
        &[AttrValue::SmallInt(3), AttrValue::Bool(true)],
        ConversionPolicy::Exact,
    )
    .unwrap();
    assert_eq!(a.as_slice(), &[3., 1.]);
    for value in [
        AttrValue::Null,
        AttrValue::Text("3".into()),
        AttrValue::Int(i64::MAX),
    ] {
        assert!(NumArray::<f64>::from_attributes(&[value], ConversionPolicy::Exact).is_err());
    }
    assert!(
        NumArray::<f64>::from_attributes(&[AttrValue::Int(i64::MAX)], ConversionPolicy::Round)
            .is_ok()
    );
}
