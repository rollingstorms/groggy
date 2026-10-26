use groggy::{types::GraphType, AttrValue, Graph};
#[test]
fn weighted_parallel_scoped_operators_keep_captured_values_direction_and_identity() {
    for directed in [false, true] {
        let mut graph = Graph::new_with_type_manual(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        });
        let nodes = graph.add_nodes(4).unwrap();
        graph.remove_node(nodes[1]).unwrap();
        let a = nodes[0];
        let b = nodes[2];
        let isolated = nodes[3];
        let first = graph.add_edge(a, b).unwrap();
        let excluded = graph.add_edge(a, b).unwrap();
        let loop_edge = graph.add_edge(a, a).unwrap();
        graph
            .set_edge_attr(first, "weight".into(), AttrValue::Float(2.5))
            .unwrap();
        graph
            .set_edge_attr(excluded, "weight".into(), AttrValue::Int(100))
            .unwrap();
        graph
            .set_edge_attr(loop_edge, "weight".into(), AttrValue::Int(3))
            .unwrap();
        let root = graph.read_view().unwrap();
        let scoped = groggy::GraphView::from_membership(
            root.read_state().clone(),
            nodes.iter().copied().filter(|&id| id != nodes[1]).collect(),
            [first, loop_edge].into_iter().collect(),
        )
        .unwrap();
        graph
            .set_edge_attr(first, "weight".into(), AttrValue::Int(999))
            .unwrap();
        graph.remove_node(a).unwrap();
        for sparse in [false, true] {
            let matrix = scoped
                .adjacency_matrix::<f64>(sparse, Some("weight"))
                .unwrap();
            assert_eq!(matrix.shape(), (3, 3));
            assert_eq!(matrix.is_sparse(), sparse);
            assert_eq!(
                matrix.to_vec().unwrap(),
                vec![
                    3.,
                    2.5,
                    0.,
                    if directed { 0. } else { 2.5 },
                    0.,
                    0.,
                    0.,
                    0.,
                    0.
                ]
            );
            assert_eq!(
                matrix.row_labels().unwrap(),
                &[
                    format!("node_{a}"),
                    format!("node_{b}"),
                    format!("node_{isolated}")
                ]
            );
            assert_eq!(matrix.column_names(), matrix.row_labels().unwrap());
        }
        assert_eq!(
            scoped
                .adjacency_matrix::<i64>(true, None)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![1, 1, 0, if directed { 0 } else { 1 }, 0, 0, 0, 0, 0]
        );
    }
}
#[test]
fn live_and_frozen_feature_exports_share_scoped_column_and_conversion_rules() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(3).unwrap();
    graph
        .set_node_attr(nodes[0], "a".into(), AttrValue::Int(7))
        .unwrap();
    graph
        .set_node_attr(nodes[1], "a".into(), AttrValue::Text("zero-imputed".into()))
        .unwrap();
    graph
        .set_node_attr(nodes[0], "b".into(), AttrValue::Bool(true))
        .unwrap();
    graph
        .set_node_attr(nodes[1], "b".into(), AttrValue::Null)
        .unwrap();
    graph
        .set_node_attr(nodes[2], "other".into(), AttrValue::Int(99))
        .unwrap();
    let root = graph.read_view().unwrap();
    let live = graph.to_matrix::<f64>().unwrap();
    let frozen = root.to_matrix::<f64>().unwrap();
    assert_eq!(live.to_vec().unwrap(), frozen.to_vec().unwrap());
    assert_eq!(live.column_names(), frozen.column_names());
    let view = root.select_nodes(&nodes[..2]).unwrap();
    let matrix = view.to_matrix::<i64>().unwrap();
    assert_eq!(matrix.column_names(), &["a".to_string(), "b".to_string()]);
    assert_eq!(matrix.to_vec().unwrap(), vec![7, 1, 0, 0]);
    graph
        .set_node_attr(nodes[0], "a".into(), AttrValue::Int(8))
        .unwrap();
    assert_eq!(
        view.to_matrix::<f64>().unwrap().to_vec().unwrap(),
        vec![7., 1., 0., 0.]
    );
    assert_eq!(matrix.get(0, 0), Some(7));
}
#[test]
fn missing_null_text_weight_policies_and_empty_shapes_match_graph_exports() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(2).unwrap();
    let first = graph.add_edge(nodes[0], nodes[1]).unwrap();
    let second = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .set_edge_attr(first, "weight".into(), AttrValue::Null)
        .unwrap();
    let view = graph.read_view().unwrap();
    assert_eq!(
        view.adjacency_matrix::<f64>(true, Some("weight"))
            .unwrap()
            .get(0, 1),
        Some(1.)
    );
    graph
        .set_edge_attr(first, "weight".into(), AttrValue::Text("1.75".into()))
        .unwrap();
    let view = graph.read_view().unwrap();
    let live = graph
        .adjacency_from_entities::<f64>(&nodes, &[first, second], true, Some("weight"))
        .unwrap();
    assert_eq!(
        view.adjacency_matrix::<f64>(true, Some("weight"))
            .unwrap()
            .to_vec()
            .unwrap(),
        live.to_vec().unwrap()
    );
    graph
        .set_edge_attr(first, "weight".into(), AttrValue::Text("invalid".into()))
        .unwrap();
    assert!(graph
        .read_view()
        .unwrap()
        .adjacency_matrix::<f64>(true, Some("weight"))
        .is_err());
    let empty = Graph::new_manual().read_view().unwrap();
    assert_eq!(empty.to_matrix::<f64>().unwrap().shape(), (0, 0));
    assert_eq!(
        empty.adjacency_matrix::<f64>(true, None).unwrap().shape(),
        (0, 0)
    );
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    assert_eq!(
        graph
            .read_view()
            .unwrap()
            .to_matrix::<f64>()
            .unwrap()
            .shape(),
        (1, 0)
    );
}

#[test]
fn explicit_numeric_text_conversion_is_independent_of_text_storage_encoding() {
    use groggy::{
        storage::matrix::FromAttrValue,
        types::{CompactString, CompressedData},
    };
    for value in [
        AttrValue::Text("17".into()),
        AttrValue::CompactText(CompactString::new("17")),
        AttrValue::CompressedText(CompressedData::compress_text("17")),
    ] {
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&value).unwrap(),
            17.
        );
        assert_eq!(
            <i64 as FromAttrValue<i64>>::from_attr_value(&value).unwrap(),
            17
        );
    }
    for value in [
        AttrValue::Text("true".into()),
        AttrValue::CompactText(CompactString::new("true")),
        AttrValue::CompressedText(CompressedData::compress_text("true")),
    ] {
        assert!(<bool as FromAttrValue<bool>>::from_attr_value(&value).unwrap());
    }
    let mut invalid = CompressedData::compress_text("17");
    invalid.data.push(0xff);
    assert!(
        <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::CompressedText(invalid)).is_err()
    );
}
