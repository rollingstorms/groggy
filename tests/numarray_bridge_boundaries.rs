use groggy::{
    storage::{
        matrix::DType,
        array::numeric_arithmetic::{attribute_inputs, integer_indices, ConversionPolicy},
        NumArray,
    },
    AttrValue,
};
#[test]
fn attribute_inference_preserves_integer_minimum_and_rejects_non_numeric_values() {
    let (dtype, values) =
        attribute_inputs(&[AttrValue::Int(i64::MIN), AttrValue::Int(i64::MAX)]).unwrap();
    assert_eq!(dtype, DType::Int64);
    assert_eq!(
        NumArray::<i64>::from_inputs(&values, ConversionPolicy::Exact)
            .unwrap()
            .as_slice(),
        &[i64::MIN, i64::MAX]
    );
    assert_eq!(
        attribute_inputs(&[AttrValue::Int(i64::from(i32::MIN))])
            .unwrap()
            .0,
        DType::Int32
    );
    assert_eq!(
        attribute_inputs(&[AttrValue::Float(1.), AttrValue::Int(i64::MAX)])
            .unwrap()
            .0,
        DType::Float64
    );
    assert!(attribute_inputs(&[AttrValue::Null]).is_err());
    assert!(attribute_inputs(&[AttrValue::Text("3".into())]).is_err());
}
#[test]
fn integer_positions_and_native_ids_never_wrap_saturate_or_truncate() {
    assert_eq!(
        integer_indices(&[i64::MIN, i64::MAX]).unwrap(),
        vec![i64::MIN, i64::MAX]
    );
    assert_eq!(integer_indices(&[-2., 0., 1.]).unwrap(), vec![-2, 0, 1]);
    for values in [[0.5], [-0.5], [f64::NAN], [f64::INFINITY], [2f64.powi(63)]] {
        assert!(integer_indices(&values).is_err());
    }
    assert_eq!(
        NumArray::<i64>::from_usize_ids(&[0, 123])
            .unwrap()
            .as_slice(),
        &[0, 123]
    );
    if usize::BITS >= 64 {
        assert!(NumArray::<i64>::from_usize_ids(&[usize::MAX]).is_err());
    }
}
