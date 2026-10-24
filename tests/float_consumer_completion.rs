use groggy::{
    storage::array::BaseArray, subgraphs::hierarchical::AggregationFunction as H,
    traits::subgraph_operations::EdgeAggregationFunction as E, AttrValue as V,
};
#[test]
fn scalar_comparisons_sorting_and_inference_share_wide_numeric_rules() {
    let a = BaseArray::new(vec![
        V::Int(9_007_199_254_740_993),
        V::Float64(9_007_199_254_740_992.),
        V::Float64(f64::NAN),
    ]);
    assert_eq!(
        a.compare_scalar(&V::Float64(9_007_199_254_740_992.), ">")
            .unwrap(),
        vec![true, false, false]
    );
    assert_eq!(
        a.compare_scalar(&V::Float64(f64::NAN), "!=").unwrap(),
        vec![true; 3]
    );
    assert_eq!(a.infer_numeric_type(), Some(groggy::NumericType::Float64));
    let a = BaseArray::new(vec![
        V::Int(i64::MAX),
        V::Float64(2f64.powi(63)),
        V::Float(0.5),
    ]);
    assert_eq!(a.sort_indices(true), vec![2, 0, 1]);
    let mixed = BaseArray::new(vec![
        V::Float(f32::NAN),
        V::Float64(0.),
        V::Float(100.),
        V::Text("a".into()),
        V::Null,
    ]);
    assert_eq!(mixed.sort_indices(true), vec![4, 1, 2, 0, 3]);
    let null = BaseArray::new(vec![V::Null]);
    assert_eq!(null.compare_scalar(&V::Null, "<").unwrap(), vec![false]);
    assert_eq!(null.compare_scalar(&V::Null, ">=").unwrap(), vec![true]);
}
#[test]
fn legacy_aggregators_keep_float64_and_checked_integer_cancellation() {
    let values = vec![V::Float64(1. + 2f64.powi(-40)), V::Float64(1e300)];
    assert_eq!(H::Max.aggregate(&values).unwrap(), V::Float64(1e300));
    assert_eq!(E::Min.aggregate(&values).unwrap(), values[0]);
    let integers = vec![V::Int(i64::MAX), V::Int(1), V::Int(-i64::MAX)];
    assert_eq!(E::Sum.aggregate(&integers).unwrap(), V::Int(1));
    assert_eq!(H::Sum.aggregate(&integers).unwrap(), V::Int(1));
    assert!(H::Sum.aggregate(&[V::Int(i64::MAX), V::Int(1)]).is_err());
    assert_eq!(
        H::Mean
            .aggregate(&[V::Float64(f64::MAX), V::Float64(f64::MAX)])
            .unwrap(),
        V::Float64(f64::MAX)
    );
}
