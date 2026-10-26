use crate::{
    AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch, GraphConfig,
};
use std::collections::HashMap;
fn initial(graph: &mut Graph) -> u64 {
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                node_attrs: HashMap::from([(
                    "v".into(),
                    vec![(N::Added(0), V::Float64Vec(vec![1.0, -0.0]))],
                )]),
                edge_attrs: HashMap::from([("v".into(), vec![(E::Added(0), V::Null)])]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "initial".into(),
            "test".into(),
        )
        .unwrap()
        .state_id
}
fn config() -> GraphConfig {
    let mut config = GraphConfig::performance_optimized();
    config.graph_type = crate::types::GraphType::Directed;
    config.query_timeout_ms = 123;
    config.worker_threads = 2;
    config.enable_compression = true;
    config.compression_level = 3;
    config
}
fn same_config(a: &GraphConfig, b: &GraphConfig) {
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}
#[test]
fn owned_historical_capture_preserves_config_storage_ids_branches_and_acceptance_clocks() {
    let mut graph = Graph::with_config_manual(config()).unwrap();
    graph.set_automatic_publication(true).unwrap();
    let first = initial(&mut graph);
    graph.create_branch("old".into()).unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "original".into())
        .unwrap();
    graph
        .writer("test")
        .set_node_external_id(0, "renamed".into())
        .unwrap();
    let inactive = graph.try_add_node().unwrap();
    graph.writer("test").remove_nodes(vec![inactive]).unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .unset_node_attr(0, "v".into())
        .unwrap();
    let latest = graph.current_state_id();
    let before = graph
        .pool()
        .get_node_attribute_column(&"v".into())
        .unwrap()
        .len();
    let capture = graph.owned_snapshot_at_commit(first).unwrap();
    let clocks = graph.publication_metadata(first).unwrap();
    assert!(capture.is_owned());
    same_config(capture.configuration().unwrap(), graph.configuration());
    graph
        .writer("test")
        .set_node_attr(0, "v".into(), V::Int(9))
        .unwrap();
    drop(graph);
    let mut copy = capture.to_graph().unwrap();
    same_config(copy.configuration(), &config());
    assert_eq!(copy.current_state_id(), first);
    assert_eq!(copy.publication_metadata(first).unwrap(), clocks);
    assert_eq!(
        copy.pool()
            .get_node_attribute_column(&"v".into())
            .unwrap()
            .len(),
        before
    );
    assert_eq!(
        copy.allocated_entity_for_external_id("original"),
        Some(crate::ExternalEntityRef::Node(0))
    );
    assert_eq!(
        copy.allocated_entity_for_external_id("renamed"),
        Some(crate::ExternalEntityRef::Node(0))
    );
    assert_eq!(copy.node_by_external_id("renamed", None), None);
    assert!(copy
        .history_records()
        .iter()
        .any(|commit| commit.id == latest));
    assert_eq!(copy.list_branches().len(), 2);
    assert_eq!(
        copy.get_node_attr(0, &"v".into()).unwrap(),
        Some(V::Float64Vec(vec![1.0, -0.0]))
    );
    assert_eq!(copy.try_add_node().unwrap(), inactive + 1);
    assert!(copy.current_state_id() > latest);
    // Restoring one copy never changes the reusable capture or another copy.
    let other = capture.to_graph().unwrap();
    assert_eq!(other.current_state_id(), first);
    assert_eq!(other.node_count(), 2);
}
#[test]
fn repeated_branch_switch_and_reset_reuse_retained_columns() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    graph.create_branch("before".into()).unwrap();
    graph
        .writer("test")
        .set_node_attr(0, "v".into(), V::Null)
        .unwrap();
    graph
        .writer("test")
        .set_edge_attr(0, "v".into(), V::Float64(-0.0))
        .unwrap();
    graph
        .writer("test")
        .unset_node_attr(1, "missing".into())
        .unwrap();
    let node_len = graph
        .pool()
        .get_node_attribute_column(&"v".into())
        .unwrap()
        .len();
    let edge_len = graph
        .pool()
        .get_edge_attribute_column(&"v".into())
        .unwrap()
        .len();
    let kind_len = graph
        .pool()
        .get_node_attribute_column(&"entity_type".into())
        .unwrap()
        .len();
    for _ in 0..100 {
        graph.checkout_branch("before".into()).unwrap();
        assert_eq!(
            graph.get_node_attr(0, &"v".into()).unwrap(),
            Some(V::Float64Vec(vec![1.0, -0.0]))
        );
        graph.checkout_branch("main".into()).unwrap();
        assert_eq!(graph.get_node_attr(0, &"v".into()).unwrap(), Some(V::Null));
        assert_eq!(
            graph.get_edge_attr(0, &"v".into()).unwrap(),
            Some(V::Float64(-0.0))
        );
        graph.reset_hard().unwrap();
    }
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"v".into())
            .unwrap()
            .len(),
        node_len
    );
    assert_eq!(
        graph
            .pool()
            .get_edge_attribute_column(&"v".into())
            .unwrap()
            .len(),
        edge_len
    );
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"entity_type".into())
            .unwrap()
            .len(),
        kind_len
    );
    // A later append updates the lazy restoration lookup as well.
    graph
        .writer("test")
        .set_node_attr(0, "v".into(), V::Float64(5.0))
        .unwrap();
    graph.checkout_branch("before".into()).unwrap();
    graph.checkout_branch("main".into()).unwrap();
    assert_eq!(
        graph
            .pool()
            .get_node_attribute_column(&"v".into())
            .unwrap()
            .len(),
        node_len + 1
    );
}
#[test]
fn ordinary_snapshot_materialization_preserves_configuration_and_frozen_groups_without_raw_capture()
{
    let mut graph = Graph::with_config_manual(config()).unwrap();
    graph.set_automatic_publication(true).unwrap();
    initial(&mut graph);
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let meta = graph.create_meta_node(group).unwrap();
    let capture = graph.snapshot_at_commit(graph.current_state_id()).unwrap();
    assert!(!capture.is_owned());
    assert!(capture.to_graph().is_err());
    let subgraph = capture.as_subgraph().unwrap();
    let owner_handle = subgraph.graph();
    let owner = owner_handle.borrow();
    same_config(owner.configuration(), &config());
    assert_eq!(
        owner
            .read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Float64Vec(vec![1.0, -0.0]))
    );
}
#[test]
fn full_owned_capture_preserves_raw_membership_and_unmapped_backing_values() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let raw = graph.pool_mut().store_subgraph(
        [0, 1].into_iter().collect(),
        [0].into_iter().collect(),
        "legacy".into(),
    );
    let raw = raw.unwrap();
    graph
        .writer("test")
        .set_node_attr(0, "source".into(), V::SubgraphRef(raw))
        .unwrap();
    let unused = graph
        .pool_mut()
        .set_attr("unmapped".into(), V::Bytes(vec![0, 255]), true);
    let capture = graph
        .owned_snapshot_at_commit(graph.current_state_id())
        .unwrap();
    drop(graph);
    let copy = capture.to_graph().unwrap();
    assert_eq!(copy.pool().get_subgraph(raw).unwrap().0.len(), 2);
    assert_eq!(
        copy.pool()
            .get_attr_by_index(&"unmapped".into(), unused, true),
        Some(&V::Bytes(vec![0, 255]))
    );
}
#[test]
fn owned_capture_rejects_dirty_manual_input_and_does_not_grant_source_write_provenance() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    assert!(graph.owned_snapshot_at_commit(0).is_err());
    let head = graph.commit("manual".into(), "test".into()).unwrap();
    let capture = graph.owned_snapshot_at_commit(head).unwrap();
    let copy = capture.to_graph().unwrap();
    let foreign = copy.snapshot_at_commit(head).unwrap();
    let original = graph.snapshot_at_commit(head).unwrap();
    // Snapshot numeric IDs match but temporal diff provenance is independently owned.
    assert!(crate::algorithms::temporal::TemporalDelta::compute(&original, &foreign).is_err());
    graph.set_node_attr(node, "v".into(), V::Int(1)).unwrap();
    assert!(graph.owned_snapshot_at_commit(head).is_err());
    assert!(!copy.has_uncommitted_changes());
}
