use groggy::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    viz::streaming::{DataSource, GraphDataSource},
    AttrValue as V, Graph,
};
use std::collections::HashSet;

fn capture() -> GraphDataSource {
    GraphTable::new(
        NodesTable::new(vec![7, 30]),
        EdgesTable::new(vec![(91, 7, 30)]),
    )
    .to_viz_data_source(false)
    .unwrap()
}

#[test]
fn identity_survives_clone_move_and_transfer_to_another_thread() {
    let source = capture();
    let key = source.get_source_id();
    let clone = source.clone();
    assert_eq!(key, clone.get_source_id());
    let moved = Box::new(clone);
    assert_eq!(key, moved.get_source_id());
    let moved = std::thread::spawn(move || moved.get_source_id())
        .join()
        .unwrap();
    assert_eq!(key, moved);
    assert_eq!(
        source.get_cache_key(0, 2),
        source.clone().get_cache_key(0, 2)
    );
}

#[test]
fn independent_captures_do_not_alias_even_when_shape_and_values_match() {
    let sources = (0..256).map(|_| capture()).collect::<Vec<_>>();
    let ids = sources
        .iter()
        .map(DataSource::get_source_id)
        .collect::<HashSet<_>>();
    assert_eq!(ids.len(), sources.len());
    assert_ne!(
        sources[0].get_cache_key(0, 2),
        sources[1].get_cache_key(0, 2)
    );
    let empty = GraphTable::empty().to_viz_data_source(false).unwrap();
    assert_ne!(
        empty.get_source_id(),
        GraphTable::empty()
            .to_viz_data_source(false)
            .unwrap()
            .get_source_id()
    );
}

#[test]
fn graph_captures_have_distinct_identity_and_shared_clones_remain_frozen() {
    let mut graph = Graph::new_manual();
    let id = graph.add_node().unwrap();
    graph
        .set_node_attr(id, "name".into(), V::Text("before".into()))
        .unwrap();
    let before = GraphDataSource::new(&graph);
    let clone = before.clone();
    graph
        .set_node_attr(id, "name".into(), V::Text("after".into()))
        .unwrap();
    let after = GraphDataSource::new(&graph);
    assert_ne!(before.get_source_id(), after.get_source_id());
    assert_eq!(before.get_source_id(), clone.get_source_id());
    assert_eq!(
        clone.node(&id.to_string()).unwrap().attributes["name"],
        V::Text("before".into())
    );
    assert_eq!(
        after.node(&id.to_string()).unwrap().attributes["name"],
        V::Text("after".into())
    );
}

#[test]
fn owned_payload_exports_cannot_mutate_the_capture_or_its_clones() {
    let source = capture();
    let clone = source.clone();
    let mut nodes = source.get_graph_nodes();
    nodes[0].id = "changed".into();
    nodes[0].attributes.insert("x".into(), V::Int(5));
    let mut edges = source.get_graph_edges();
    edges[0].target = "changed".into();
    assert_eq!(source.get_graph_nodes()[0].id, "7");
    assert!(clone.get_graph_nodes()[0].attributes.is_empty());
    assert_eq!(clone.get_graph_edges()[0].target, "30");
    assert_eq!(source.get_source_id(), clone.get_source_id());
}
