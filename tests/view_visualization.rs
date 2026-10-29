use groggy::{
    viz::streaming::{DataSource, GraphDataSource},
    AttrValue, Graph,
};
#[test]
fn frozen_visualization_preserves_selected_parallel_instances_typed_values_and_direction() {
    for directed in [false, true] {
        let mut graph = if directed {
            Graph::new_directed()
        } else {
            Graph::new()
        };
        graph.add_nodes(5).unwrap();
        for (s, t) in [(1, 3), (1, 3), (3, 3), (0, 4)] {
            graph.add_edge(s, t).unwrap();
        }
        graph
            .set_node_attr(1, "name".into(), AttrValue::Text("selected".into()))
            .unwrap();
        let wide = 1. + 2f64.powi(-40);
        graph
            .set_node_attr(
                1,
                "position".into(),
                AttrValue::Float64Vec(vec![wide, 1e100]),
            )
            .unwrap();
        graph
            .set_edge_attr(1, "weight".into(), AttrValue::Float64(wide))
            .unwrap();
        let selected = graph.read_view().unwrap().select_edges(&[1, 2]).unwrap();
        graph.remove_node(1).unwrap();
        drop(graph);
        let source = GraphDataSource::from_view(&selected).unwrap();
        let metadata = source.get_graph_metadata();
        assert_eq!(
            (
                metadata.node_count,
                metadata.edge_count,
                metadata.is_directed
            ),
            (2, 2, directed)
        );
        let nodes = source.get_graph_nodes();
        assert_eq!(
            nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
            vec!["1", "3"]
        );
        assert_eq!(nodes[0].label.as_deref(), Some("selected"));
        assert_eq!(nodes[0].position.as_ref().unwrap().x, wide);
        assert_eq!(nodes[0].position.as_ref().unwrap().y, 1e100);
        let edges = source.get_graph_edges();
        assert_eq!(
            edges.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["1", "2"]
        );
        assert_eq!(edges[0].weight, Some(wide));
        let key = source.get_source_id();
        assert_eq!(source.clone().get_source_id(), key);
        assert_ne!(
            GraphDataSource::from_view(&selected)
                .unwrap()
                .get_source_id(),
            key
        );
        std::thread::spawn(move || assert_eq!(source.get_graph_nodes().len(), 2))
            .join()
            .unwrap();
    }
}
#[test]
fn empty_and_isolated_captures_keep_rows_without_implicit_edge_induction() {
    let mut graph = Graph::new();
    let empty = GraphDataSource::from_view(&graph.read_view().unwrap()).unwrap();
    assert_eq!(empty.total_rows(), 0);
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    let root = graph.read_view().unwrap();
    let isolated = groggy::GraphView::from_membership(
        root.read_state().clone(),
        [0, 1].into(),
        Default::default(),
    )
    .unwrap();
    let source = GraphDataSource::from_view(&isolated).unwrap();
    assert_eq!(source.get_graph_nodes().len(), 2);
    assert!(source.get_graph_edges().is_empty());
}
