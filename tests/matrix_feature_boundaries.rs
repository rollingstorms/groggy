use groggy::{
    storage::{
        matrix::FromAttrValue,
        matrix::{unified_matrix::MatrixStorage, Strides, UnifiedMatrix},
    },
    types::AttrValue,
    Graph,
};
#[test]
fn feature_export_reads_branch_references_and_sorts_columns() {
    let mut graph = Graph::new_manual();
    let a = graph.add_node().unwrap();
    let b = graph.add_node().unwrap();
    graph
        .set_node_attr(a, "z".into(), AttrValue::Int(i64::MAX))
        .unwrap();
    graph
        .set_node_attr(a, "a".into(), AttrValue::Int(4))
        .unwrap();
    graph
        .set_node_attr(b, "a".into(), AttrValue::Text("not numeric".into()))
        .unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    graph.create_branch("original".into()).unwrap();
    graph
        .set_node_attr(a, "a".into(), AttrValue::Int(99))
        .unwrap();
    graph.commit("changed".into(), "test".into()).unwrap();
    assert_eq!(graph.to_matrix_i64().unwrap().get(0, 0), Some(99));
    graph.checkout_branch("original".into()).unwrap();
    let matrix = graph.to_matrix_i64().unwrap();
    assert_eq!(matrix.column_names(), ["a", "z"]);
    assert_eq!(
        matrix.row_labels().unwrap(),
        [format!("node_{a}"), format!("node_{b}")]
    );
    assert_eq!(matrix.to_vec().unwrap(), vec![4, i64::MAX, 0, 0]);
    let empty = Graph::new_manual().to_matrix_i64().unwrap();
    assert_eq!(empty.shape(), (0, 0));
    assert!(empty.row_labels().unwrap().is_empty());
}
#[test]
fn explicit_attribute_converters_reject_range_failure_preserve_legacy_imputation() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2f32.powi(63)] {
        assert!(<i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Float(value)).is_err());
    }
    for value in [i64::MIN, i64::MAX, i64::from(i32::MAX) + 1] {
        assert!(<i32 as FromAttrValue<i32>>::from_attr_value(&AttrValue::Int(value)).is_err());
    }
    assert_eq!(
        <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Float(-3.5)).unwrap(),
        -3
    );
    assert_eq!(
        <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Null).unwrap(),
        0
    );
    assert_eq!(
        <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Int(i64::MAX)).unwrap(),
        i64::MAX
    );
    assert!(
        <f32 as FromAttrValue<f32>>::from_attr_value(&AttrValue::Text("1e300".into())).is_err()
    );
    assert!(
        <f32 as FromAttrValue<f32>>::from_attr_value(&AttrValue::Text("inf".into()))
            .unwrap()
            .is_infinite()
    );
}
#[test]
fn fill_respects_logical_addresses_and_validates_before_mutation() {
    let mut matrix = UnifiedMatrix::from_data(vec![1i64, 2, 3, 4, 5, 6], 2, 3).unwrap();
    matrix.shape.cols = 2;
    matrix.strides = Strides {
        row_stride: 3,
        col_stride: 1,
    };
    matrix.fill(9).unwrap();
    if let MatrixStorage::Dense(buffer) = &matrix.storage {
        assert_eq!(buffer.data().unwrap().as_slice(), &[9, 9, 3, 9, 9, 6]);
    } else {
        panic!();
    }
    matrix.strides.row_stride = usize::MAX;
    assert!(matrix.fill(0).is_err());
    if let MatrixStorage::Dense(buffer) = &matrix.storage {
        assert_eq!(buffer.data().unwrap().as_slice(), &[9, 9, 3, 9, 9, 6]);
    }
    let mut sparse = UnifiedMatrix::from_csr(2, 3, vec![2i64], vec![1], vec![0, 1, 1]).unwrap();
    sparse.fill(4).unwrap();
    assert_eq!(sparse.to_vec().unwrap(), vec![4; 6]);
    assert!(matches!(sparse.storage, MatrixStorage::Sparse(_)));
    sparse.fill(0).unwrap();
    if let MatrixStorage::Sparse(storage) = sparse.storage {
        assert!(storage.values.is_empty());
    }
}
