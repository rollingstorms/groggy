use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use std::sync::Arc;
#[test]
fn selected_copy_preserves_ids_values_direction_and_publishes_one_independent_base() {
    let mut original = Graph::new_directed();
    original.add_nodes(5).unwrap();
    original.add_edge(1, 3).unwrap();
    original.add_edge(1, 3).unwrap();
    original
        .set_node_attr(1, "score".into(), AttrValue::Float64(0.123456789012345))
        .unwrap();
    original
        .set_edge_attr(1, "refs".into(), AttrValue::NodeArray(vec![3, 1]))
        .unwrap();
    let owner = Arc::new(OwnerCell::new(original));
    let selected = Subgraph::new(owner.clone(), [1, 3, 4].into(), [1].into(), "copy".into());
    let frozen = selected.pin_view().unwrap();
    owner.borrow_mut().remove_node(1).unwrap();
    assert!(selected.pin_view().is_err());
    drop(selected);
    drop(owner);
    let mut copy = frozen.to_graph().unwrap();
    assert_eq!(copy.node_count(), 3);
    assert_eq!(copy.edge_count(), 1);
    assert!(copy.is_directed());
    assert_eq!(copy.current_state_id(), 1);
    assert!(!copy.has_uncommitted_changes());
    assert_eq!(copy.edge_endpoints(1).unwrap(), (1, 3));
    assert_eq!(
        copy.get_node_attr(1, &"score".into()).unwrap(),
        Some(AttrValue::Float64(0.123456789012345))
    );
    copy.set_node_attr(1, "score".into(), AttrValue::Int(9))
        .unwrap();
    assert_eq!(copy.current_state_id(), 2);
    assert_eq!(
        frozen.node_attr(1, &"score".into()).unwrap(),
        Some(AttrValue::Float64(0.123456789012345))
    );
    assert!(copy.add_node().unwrap() > 4);
}
#[test]
fn empty_copies_publish_no_state_and_external_local_refs_reject() {
    let mut graph = Graph::new();
    assert_eq!(
        graph
            .read_view()
            .unwrap()
            .to_graph()
            .unwrap()
            .current_state_id(),
        0
    );
    graph.add_nodes(2).unwrap();
    graph
        .set_node_attr(0, "refs".into(), AttrValue::NodeArray(vec![1]))
        .unwrap();
    let narrow = graph.read_view().unwrap().select_nodes(&[0]).unwrap();
    assert!(narrow.to_graph().is_err());
    assert_eq!(graph.node_count(), 2);
}
#[test]
fn copied_meta_sources_remain_source_qualified_after_owner_drop_and_round_trip() {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    let source = graph.read_view().unwrap().select_nodes(&[0]).unwrap();
    let group = graph.retain_group(&source, "source".into()).unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    let selected = graph.read_view().unwrap().select_nodes(&[meta]).unwrap();
    drop(graph);
    let copy = selected.to_graph().unwrap();
    assert_eq!(copy.node_count(), 1);
    assert_eq!(
        copy.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_ids(),
        vec![0]
    );
    copy.check_integrity().unwrap();
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("selected.groggy");
    copy.save_archive(&path).unwrap();
    let reopened = Graph::load_archive(&path).unwrap();
    assert_eq!(
        reopened
            .read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_ids(),
        vec![0]
    );
}
