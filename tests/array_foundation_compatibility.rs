use groggy::{
    storage::{
        array::{
            base::BaseArray as Legacy, numeric_arithmetic as legacy_numeric, BaseArray, NumArray,
        },
        numeric,
    },
    types::{AttrValue, AttrValueType},
};
use std::{any::TypeId, sync::Arc};
#[test]
fn legacy_metadata_adapter_uses_the_same_column_and_copy_on_write() {
    let column = BaseArray::new(vec![AttrValue::Int(1), AttrValue::Int(2)]);
    let mut old = Legacy::from_column(column.clone(), AttrValueType::Int);
    old.set_name("numbers".into());
    assert!(Arc::ptr_eq(&column.inner, &old.as_column().inner));
    let mut sibling = old.clone();
    sibling.set(0, AttrValue::Int(9)).unwrap();
    assert_eq!(column.data(), &vec![AttrValue::Int(1), AttrValue::Int(2)]);
    assert_eq!(old.data(), column.data());
    assert_eq!(sibling.data()[0], AttrValue::Int(9));
    assert_eq!(sibling.name().unwrap(), "numbers");
    assert_eq!(sibling.dtype(), &AttrValueType::Int);
    old.data_mut().push(AttrValue::Int(3));
    assert_eq!(column.len(), 2);
    assert_eq!(old.data().len(), 3);
    assert_eq!(old.into_column().len(), 3);
}
#[test]
fn numeric_compatibility_paths_share_types_policies_and_results() {
    assert_eq!(
        TypeId::of::<numeric::DType>(),
        TypeId::of::<groggy::storage::matrix::numeric_type::DType>()
    );
    assert_eq!(
        TypeId::of::<numeric::ConversionPolicy>(),
        TypeId::of::<legacy_numeric::ConversionPolicy>()
    );
    assert_eq!(
        TypeId::of::<numeric::NumericInput>(),
        TypeId::of::<legacy_numeric::NumericInput>()
    );
    let input = numeric::NumericInput::Integer(i64::MAX as i128);
    assert!(numeric::scalar::convert::<f64>(input, numeric::ConversionPolicy::Exact).is_err());
    assert!(legacy_numeric::convert::<f64>(input, numeric::ConversionPolicy::Exact).is_err());
    assert_eq!(numeric::reductions::mean(&[1i64, 2, 3]), Some(2.));
    assert_eq!(
        groggy::storage::array::numeric_reductions::mean(&[1i64, 2, 3]),
        Some(2.)
    );
    assert_eq!(
        NumArray::new(vec![1e308, 1., -1e308])
            .dot(&NumArray::new(vec![1., 1., 1.]))
            .unwrap(),
        1.
    );
}
#[test]
fn legacy_array_constructors_selection_reductions_and_mutation_remain_compatible() {
    let mut old = Legacy::with_name(
        vec![AttrValue::Int(4), AttrValue::Int(2), AttrValue::Int(4)],
        AttrValueType::Int,
        "x".into(),
    );
    assert_eq!(old.sort_indices(true), vec![1, 0, 2]);
    assert_eq!(
        old.take_indices(&[2, 0]).unwrap().data(),
        &vec![AttrValue::Int(4); 2]
    );
    assert_eq!(old.sum().unwrap(), AttrValue::Int(10));
    assert_eq!(old.mean().unwrap(), 10. / 3.);
    assert!(old.set(9, AttrValue::Int(0)).is_err());
    assert_eq!(
        old.filter_by_mask(&[false, true, false]).unwrap().data(),
        &vec![AttrValue::Int(2)]
    );
    assert_eq!(
        old.slice(1, 3).data(),
        &vec![AttrValue::Int(2), AttrValue::Int(4)]
    );
}

#[test]
fn canonical_and_legacy_selection_validate_whole_inputs_and_share_id_checks() {
    let a = BaseArray::new(vec![AttrValue::Int(0), AttrValue::Int(2)]);
    let legacy = Legacy::from_column(a.clone(), AttrValueType::Int);
    for ids in [&[0, 2][..], &[usize::MAX][..]] {
        assert!(a.take_indices(ids).is_err());
        assert!(legacy.take_indices(ids).is_err());
    }
    for mask in [&[true][..], &[true, false, true][..]] {
        assert!(a.filter_by_mask(mask).is_err());
        assert!(legacy.filter_by_mask(mask).is_err());
    }
    assert_eq!(a.slice(usize::MAX, usize::MAX).len(), 0);
    assert_eq!(a.slice(1, usize::MAX).data(), &vec![AttrValue::Int(2)]);
    assert_eq!(legacy.slice(1, 2).data(), a.slice(1, 1).data());
    let invalid = BaseArray::new(vec![AttrValue::Int(-1), AttrValue::Int(2)]);
    let old = Legacy::from_column(invalid.clone(), AttrValueType::Int);
    assert!(invalid.as_node_ids().is_err());
    assert!(old.as_node_ids().is_err());
    assert!(old.as_edge_ids().is_err());
    assert_eq!(old.as_node_ids_filtered(), (vec![2], vec![1]));
    assert_eq!(
        BaseArray::try_from_node_ids(vec![0, 2]).unwrap().data(),
        a.data()
    );
    assert_eq!(
        Legacy::try_from_edge_ids(vec![0, 2]).unwrap().data(),
        a.data()
    );
    if usize::BITS > 63 {
        assert!(BaseArray::try_from_node_ids(vec![usize::MAX]).is_err());
        assert!(Legacy::try_from_edge_ids(vec![usize::MAX]).is_err());
    }
}
