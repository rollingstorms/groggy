use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable},
    },
    viz::streaming::{data_source::LayoutAlgorithm, DataSource, GraphDataSource},
    AttrValue as V, Graph,
};

fn base(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    let order = columns.iter().map(|(name, _)| name.to_string()).collect();
    BaseTable::with_column_order(
        columns
            .into_iter()
            .map(|(name, values)| (name.to_string(), BaseArray::from_attr_values(values)))
            .collect(),
        order,
    )
    .unwrap()
}
fn fixture() -> GraphTable {
    GraphTable::new(
        NodesTable::from_base_table(base(vec![
            ("node_id", vec![V::SmallInt(70), V::Int(12)]),
            (
                "name",
                vec![V::Text("seventy".into()), V::Text("twelve".into())],
            ),
            ("value", vec![V::Float64(1.23456789012345), V::Null]),
            ("refs", vec![V::NodeArray(vec![12]), V::NodeArray(vec![70])]),
            ("x", vec![V::Float64(3.5), V::Float64(1.0)]),
            ("y", vec![V::Float64(7.0), V::Float64(2.0)]),
        ]))
        .unwrap(),
        EdgesTable::from_base_table(base(vec![
            ("edge_id", vec![V::SmallInt(900), V::Int(200), V::Int(300)]),
            ("source", vec![V::Int(70), V::Int(70), V::Int(12)]),
            ("target", vec![V::Int(12), V::Int(12), V::Int(12)]),
            ("weight", vec![V::SmallInt(2), V::Float64(3.75), V::Null]),
            ("payload", vec![V::Json("{\"x\": 1}".into()); 3]),
        ]))
        .unwrap(),
    )
}

#[test]
fn sparse_ids_parallel_edges_loops_and_typed_properties_survive_without_remapping() {
    let table = fixture();
    let source = table.to_viz_data_source(true).unwrap();
    let nodes = source.get_graph_nodes();
    assert_eq!(
        nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
        vec!["70", "12"]
    );
    assert_eq!(nodes[0].label.as_deref(), Some("seventy"));
    assert_eq!(nodes[0].attributes["value"], V::Float64(1.23456789012345));
    assert_eq!(nodes[1].attributes["value"], V::Null);
    assert_eq!(nodes[0].attributes["refs"], V::NodeArray(vec![12]));
    assert_eq!(nodes[0].position.as_ref().unwrap().x, 3.5);
    let edges = source.get_graph_edges();
    assert_eq!(
        edges.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
        vec!["900", "200", "300"]
    );
    assert_eq!(
        (&edges[0].source, &edges[0].target),
        (&"70".into(), &"12".into())
    );
    assert_eq!(edges[0].weight, Some(2.0));
    assert_eq!(edges[1].weight, Some(3.75));
    assert_eq!(edges[2].source, edges[2].target);
    assert_eq!(edges[0].attributes["payload"], V::Json("{\"x\": 1}".into()));
    let metadata = source.get_graph_metadata();
    assert_eq!((metadata.node_count, metadata.edge_count), (2, 3));
    assert!(metadata.is_directed && metadata.has_weights);
    assert!(
        !table
            .to_viz_data_source(false)
            .unwrap()
            .get_graph_metadata()
            .is_directed
    );
}

#[test]
fn component_adapters_keep_isolated_nodes_and_infer_only_edge_endpoints() {
    let table = fixture();
    let nodes = table.nodes().to_viz_data_source().unwrap();
    assert_eq!(nodes.get_graph_nodes().len(), 2);
    assert!(nodes.get_graph_edges().is_empty());
    let edges = table.edges().to_viz_data_source(true).unwrap();
    assert_eq!(
        edges
            .get_graph_nodes()
            .iter()
            .map(|n| n.id.as_str())
            .collect::<Vec<_>>(),
        vec!["12", "70"]
    );
    assert!(edges
        .get_graph_nodes()
        .iter()
        .all(|n| n.attributes.is_empty()));
    assert_eq!(edges.get_graph_edges().len(), 3);
}

#[test]
fn duplicate_node_or_edge_identity_rejects_the_whole_capture() {
    let table = fixture();
    assert!(NodesTable::new(vec![12, 12])
        .to_viz_data_source()
        .unwrap_err()
        .to_string()
        .contains("Duplicate"));
    let duplicate = EdgesTable::new(vec![(9, 12, 70), (9, 70, 12)]);
    assert!(duplicate
        .to_viz_data_source(false)
        .unwrap_err()
        .to_string()
        .contains("Duplicate"));
    assert!(
        GraphTable::new(NodesTable::new(vec![12, 12]), table.edges().clone())
            .to_viz_data_source(false)
            .is_err()
    );
}

#[test]
fn invalid_structural_cells_reject_instead_of_casting_or_disappearing() {
    for bad in [
        V::Null,
        V::Int(-1),
        V::Float64(1.0),
        V::Bool(true),
        V::Text("1".into()),
    ] {
        let nodes =
            NodesTable::from_base_table(base(vec![("node_id", vec![bad.clone()])])).unwrap();
        assert!(nodes.to_viz_data_source().is_err());
        for name in ["edge_id", "source", "target"] {
            let edges = EdgesTable::from_base_table(base(
                ["edge_id", "source", "target"]
                    .into_iter()
                    .map(|n| (n, vec![if n == name { bad.clone() } else { V::Int(1) }]))
                    .collect(),
            ))
            .unwrap();
            assert!(edges.to_viz_data_source(false).is_err());
        }
    }
}

#[test]
fn combined_table_rejects_dangling_endpoints_instead_of_inventing_nodes() {
    let table = GraphTable::new(NodesTable::new(vec![70]), fixture().edges().clone());
    let error = table.to_viz_data_source(false).unwrap_err().to_string();
    assert!(error.contains("endpoint outside"), "{error}");
}

#[test]
fn snapshots_are_owned_and_do_not_follow_later_source_writes() {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(2).unwrap();
    graph
        .set_node_attr(1, "name".into(), V::Text("before".into()))
        .unwrap();
    let table = graph.table().unwrap();
    let source = table.to_viz_data_source(false).unwrap();
    graph
        .set_node_attr(1, "name".into(), V::Text("after".into()))
        .unwrap();
    assert_eq!(
        source.get_graph_nodes()[1].attributes["name"],
        V::Text("before".into())
    );
    assert_eq!(source.get_graph_nodes().len(), 2);
}

#[test]
fn empty_and_overflowing_windows_are_safe_and_layout_preserves_original_ids() {
    let empty = GraphTable::empty().to_viz_data_source(false).unwrap();
    assert!(empty.get_window(usize::MAX, usize::MAX).rows.is_empty());
    let source = fixture().to_viz_data_source(false).unwrap();
    assert_eq!(source.get_window(1, usize::MAX).rows.len(), 1);
    assert!(source.get_window(3, 0).rows.is_empty());
    assert!(source.get_window(usize::MAX, 5).rows.is_empty());
    let positions = source.compute_layout(LayoutAlgorithm::Grid {
        columns: 2,
        cell_size: 1.0,
    });
    assert_eq!(
        positions
            .iter()
            .map(|p| p.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["70", "12"]
    );
    let nodes = source.get_graph_nodes();
    let mut edges = source.get_graph_edges();
    edges[0].source = "missing".into();
    assert!(GraphDataSource::from_entities(nodes, edges, false).is_err());
}
