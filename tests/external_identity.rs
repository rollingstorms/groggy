use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::{
    AttrValue, BatchNodeRef, EdgeUpsert, ExternalEntityRef, Graph, GraphBatch, NodeUpsert,
    EXTERNAL_ID,
};
use std::collections::{HashMap, HashSet};

fn node(id: &str, score: i64) -> NodeUpsert {
    NodeUpsert {
        external_id: id.into(),
        attrs: HashMap::from([("score".into(), AttrValue::Int(score))]),
    }
}

fn edge(id: &str, source: usize, target: usize, weight: i64) -> EdgeUpsert {
    EdgeUpsert {
        external_id: id.into(),
        source,
        target,
        attrs: HashMap::from([("weight".into(), AttrValue::Int(weight))]),
    }
}

#[test]
fn upserts_preserve_identity_order_properties_and_clean_idempotence() {
    let mut graph = Graph::new();
    graph.add_nodes(3);
    let ids = graph
        .upsert_nodes(vec![node("alice", 1), node("bob", 2)])
        .unwrap();
    assert_eq!(ids, vec![3, 4]);
    graph
        .set_node_attr(ids[0], "keep".into(), AttrValue::Bool(true))
        .unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    assert_eq!(
        graph
            .upsert_nodes(vec![node("bob", 2), node("alice", 1)])
            .unwrap(),
        vec![4, 3]
    );
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.node_by_external_id("alice", None), Some(3));
    assert_eq!(graph.upsert_nodes(vec![node("alice", 9)]).unwrap(), vec![3]);
    assert_eq!(
        graph.get_node_attr(3, &"keep".into()).unwrap(),
        Some(AttrValue::Bool(true))
    );
    assert_eq!(
        graph.get_node_attr(3, &"score".into()).unwrap(),
        Some(AttrValue::Int(9))
    );
}

#[test]
fn all_conflicting_node_rows_are_rejected_before_any_write_or_allocation() {
    let mut graph = Graph::new();
    let ids = graph
        .upsert_nodes(vec![node("alice", 1), node("bob", 2)])
        .unwrap();
    graph
        .upsert_edges(vec![edge("fact", ids[0], ids[1], 1)])
        .unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    let mut reserved = node("other", 5);
    reserved
        .attrs
        .insert(EXTERNAL_ID.into(), AttrValue::Text("bypass".into()));
    let error = graph
        .upsert_nodes(vec![
            node("alice", 7),
            node("new", 1),
            node("new", 2),
            node("fact", 3),
            node("", 4),
            reserved,
        ])
        .unwrap_err()
        .to_string();
    for part in [
        "2: duplicate",
        "3: external ID",
        "4: external ID",
        "5.external_id",
    ] {
        assert!(error.contains(part), "{error}");
    }
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(
        graph.get_node_attr(ids[0], &"score".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
    assert_eq!(graph.node_by_external_id("new", None), None);
    assert_eq!(graph.add_node(), 2);
}

#[test]
fn edge_upserts_preserve_parallel_facts_and_reject_endpoint_changes_atomically() {
    let mut graph = Graph::new_directed();
    let nodes = graph
        .upsert_nodes(vec![node("a", 1), node("b", 2), node("c", 3)])
        .unwrap();
    let edges = graph
        .upsert_edges(vec![
            edge("one", nodes[0], nodes[1], 1),
            edge("two", nodes[0], nodes[1], 2),
        ])
        .unwrap();
    assert_ne!(edges[0], edges[1]);
    graph.commit("initial".into(), "test".into()).unwrap();
    assert_eq!(
        graph
            .upsert_edges(vec![edge("two", nodes[0], nodes[1], 2)])
            .unwrap(),
        vec![edges[1]]
    );
    assert!(!graph.has_uncommitted_changes());
    let error = graph
        .upsert_edges(vec![
            edge("new", nodes[0], nodes[1], 3),
            edge("one", nodes[0], nodes[2], 4),
            edge("missing", nodes[0], 99, 5),
            edge("a", nodes[0], nodes[1], 6),
        ])
        .unwrap_err()
        .to_string();
    for part in ["1: edge", "2: endpoints", "3: external ID"] {
        assert!(error.contains(part), "{error}");
    }
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(
        graph.edge_endpoints(edges[0]).unwrap(),
        (nodes[0], nodes[1])
    );
    assert_eq!(
        graph.get_edge_attr(edges[0], &"weight".into()).unwrap(),
        Some(AttrValue::Int(1))
    );
    assert_eq!(graph.add_edge(nodes[0], nodes[1]).unwrap(), 2);
}

#[test]
fn undirected_upserts_accept_reversed_endpoints_without_rewiring() {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(2);
    let first = graph
        .upsert_edges(vec![edge("fact", nodes[0], nodes[1], 1)])
        .unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    assert_eq!(
        graph
            .upsert_edges(vec![edge("fact", nodes[1], nodes[0], 1)])
            .unwrap(),
        first
    );
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(
        graph.edge_endpoints(first[0]).unwrap(),
        (nodes[0], nodes[1])
    );
}

#[test]
fn rename_and_branch_switch_preserve_current_names_and_allocation_reservations() {
    let mut graph = Graph::new();
    let id = graph.upsert_nodes(vec![node("old", 1)]).unwrap()[0];
    let first = graph.commit("initial".into(), "test".into()).unwrap();
    graph.create_branch("original".into()).unwrap();
    graph.set_node_external_id(id, "new".into()).unwrap();
    let second = graph.commit("renamed".into(), "test".into()).unwrap();
    assert_eq!(graph.node_by_external_id("old", None), None);
    assert_eq!(graph.node_by_external_id("new", None), Some(id));
    assert_eq!(
        graph
            .snapshot_at_commit(first)
            .unwrap()
            .node_by_external_id("old", None),
        Some(id)
    );
    assert_eq!(
        graph
            .snapshot_at_commit(second)
            .unwrap()
            .node_by_external_id("new", None),
        Some(id)
    );
    graph.checkout_branch("original".into()).unwrap();
    assert_eq!(graph.node_by_external_id("old", None), Some(id));
    assert_eq!(graph.node_by_external_id("new", None), None);
    assert_eq!(
        graph.allocated_entity_for_external_id("new"),
        Some(ExternalEntityRef::Node(id))
    );
    assert!(graph.upsert_nodes(vec![node("new", 5)]).is_err());
    assert!(!graph.has_uncommitted_changes());
    graph.remove_node(id).unwrap();
    graph.commit("removed".into(), "test".into()).unwrap();
    assert_eq!(graph.node_by_external_id("old", None), None);
    assert!(graph.upsert_nodes(vec![node("old", 6)]).is_err());
    assert_eq!(
        graph.allocated_entity_for_external_id("old"),
        Some(ExternalEntityRef::Node(id))
    );
}

#[test]
fn group_filters_do_not_create_separate_uuid_namespaces() {
    let mut graph = Graph::new();
    let mut row = node("uuid", 1);
    row.attrs
        .insert("group_id".into(), AttrValue::Text("g1".into()));
    let id = graph.upsert_nodes(vec![row]).unwrap()[0];
    assert_eq!(graph.node_by_external_id("uuid", Some("g1")), Some(id));
    assert_eq!(graph.node_by_external_id("uuid", Some("g2")), None);
    let other = graph.add_node();
    assert!(graph.set_node_external_id(other, "uuid".into()).is_err());
    let edge = graph.add_edge(id, other).unwrap();
    assert!(graph.set_edge_external_id(edge, "uuid".into()).is_err());
    graph.set_edge_external_id(edge, "edge".into()).unwrap();
    graph
        .set_edge_attr(edge, "group_id".into(), AttrValue::Text("g1".into()))
        .unwrap();
    assert_eq!(graph.edge_by_external_id("edge", Some("g1")), Some(edge));
    assert_eq!(graph.edge_by_external_id("edge", Some("g2")), None);
}

#[test]
fn managed_identity_cannot_be_bypassed_by_scalar_bulk_column_or_batch_writes() {
    let mut graph = Graph::new();
    let node = graph.add_node();
    let edge = graph.add_edge(node, node).unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    let value = AttrValue::Text("bypass".into());
    assert!(graph
        .set_node_attr(node, EXTERNAL_ID.into(), value.clone())
        .is_err());
    assert!(graph
        .set_edge_attr(edge, EXTERNAL_ID.into(), value.clone())
        .is_err());
    assert!(graph
        .set_node_attrs(HashMap::from([(
            EXTERNAL_ID.into(),
            vec![(node, value.clone())]
        )]))
        .is_err());
    assert!(graph
        .set_edge_attrs(HashMap::from([(
            EXTERNAL_ID.into(),
            vec![(edge, value.clone())]
        )]))
        .is_err());
    assert!(graph
        .set_node_attr_column(EXTERNAL_ID.into(), vec![(node, value.clone())])
        .is_err());
    let mut batch = GraphBatch {
        add_nodes: 1,
        ..Default::default()
    };
    batch
        .node_attrs
        .insert(EXTERNAL_ID.into(), vec![(BatchNodeRef::Added(0), value)]);
    assert!(graph
        .apply_batch(batch, graph.current_state_id(), "bad".into(), "test".into())
        .is_err());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.add_node(), 1);
}

#[test]
fn subgraph_bulk_writes_use_validation_and_change_tracking() {
    let mut graph = Graph::new();
    let ids = graph
        .upsert_nodes(vec![node("a", 1), node("b", 2)])
        .unwrap();
    let edge = graph
        .upsert_edges(vec![edge("fact", ids[0], ids[1], 1)])
        .unwrap()[0];
    let first = graph.commit("initial".into(), "test".into()).unwrap();
    let graph = graph.into_shared();
    let view = Subgraph::new(
        graph.clone(),
        HashSet::from([ids[0], ids[1]]),
        HashSet::from([edge]),
        "test".into(),
    );
    assert!(view
        .set_node_attrs(HashMap::from([(
            EXTERNAL_ID.into(),
            vec![(ids[0], AttrValue::Text("bypass".into()))]
        )]))
        .is_err());
    assert!(view
        .set_edge_attrs(HashMap::from([(
            EXTERNAL_ID.into(),
            vec![(edge, AttrValue::Text("bypass".into()))]
        )]))
        .is_err());
    assert!(!graph.borrow().has_uncommitted_changes());
    view.set_node_attrs(HashMap::from([(
        "score".into(),
        vec![(ids[0], AttrValue::Int(9))],
    )]))
    .unwrap();
    view.set_edge_attrs(HashMap::from([(
        "weight".into(),
        vec![(edge, AttrValue::Int(8))],
    )]))
    .unwrap();
    let second = graph
        .borrow_mut()
        .commit("view write".into(), "test".into())
        .unwrap();
    assert_eq!(
        graph
            .borrow()
            .snapshot_at_commit(first)
            .unwrap()
            .node_attr(ids[0], &"score".into()),
        Some(AttrValue::Int(1))
    );
    assert_eq!(
        graph
            .borrow()
            .snapshot_at_commit(second)
            .unwrap()
            .node_attr(ids[0], &"score".into()),
        Some(AttrValue::Int(9))
    );
    assert_eq!(
        graph
            .borrow()
            .snapshot_at_commit(second)
            .unwrap()
            .edge_attr(edge, &"weight".into()),
        Some(AttrValue::Int(8))
    );
}

#[test]
fn snapshot_import_rebuilds_bindings_and_rejects_cross_kind_duplicates() {
    let mut snapshot = groggy::state::state::GraphSnapshot::empty(0);
    snapshot.active_nodes = vec![4];
    snapshot.node_attributes.insert(
        4,
        HashMap::from([(EXTERNAL_ID.into(), AttrValue::Text("uuid".into()))]),
    );
    let graph = Graph::from_snapshot(snapshot.clone()).unwrap();
    assert_eq!(graph.node_by_external_id("uuid", None), Some(4));
    snapshot.edges.insert(6, (4, 4));
    snapshot.edge_attributes.insert(
        6,
        HashMap::from([(EXTERNAL_ID.into(), AttrValue::Text("uuid".into()))]),
    );
    assert!(Graph::from_snapshot(snapshot).is_err());
}

#[test]
fn live_export_import_preserves_current_managed_ids() {
    let mut graph = Graph::new_directed();
    let padding = graph.add_nodes(3);
    graph.remove_nodes(&padding).unwrap();
    let ids = graph
        .upsert_nodes(vec![node("a", 1), node("b", 2)])
        .unwrap();
    let removed_edge = graph.add_edge(ids[0], ids[1]).unwrap();
    graph.remove_edge(removed_edge).unwrap();
    graph
        .upsert_edges(vec![edge("fact", ids[0], ids[1], 3)])
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("graph.json");
    graph.save_to_path(&path).unwrap();
    let loaded = Graph::load_from_path(&path).unwrap();
    assert_eq!(loaded.node_by_external_id("a", None), Some(ids[0]));
    assert_eq!(loaded.node_by_external_id("b", None), Some(ids[1]));
    assert_eq!(loaded.edge_by_external_id("fact", None), Some(1));
}

#[test]
fn compressed_external_ids_remain_addressable_and_reconstructible() {
    let mut graph = Graph::new();
    let name = "external-id-".repeat(1000);
    let id = graph.upsert_nodes(vec![node(&name, 1)]).unwrap()[0];
    assert_eq!(graph.node_by_external_id(&name, None), Some(id));
    let state = graph.commit("long ID".into(), "test".into()).unwrap();
    let snapshot = graph.snapshot_at_commit(state).unwrap();
    assert_eq!(snapshot.node_by_external_id(&name, None), Some(id));
    assert_eq!(graph.upsert_nodes(vec![node(&name, 1)]).unwrap(), vec![id]);
    assert!(!graph.has_uncommitted_changes());
    snapshot.as_subgraph().unwrap();
}

#[test]
fn reset_keeps_uncommitted_identity_reservations_with_allocated_ids() {
    let mut graph = Graph::new();
    graph.add_node();
    graph.commit("initial".into(), "test".into()).unwrap();
    let transient = graph.upsert_nodes(vec![node("transient", 1)]).unwrap()[0];
    graph.reset_hard().unwrap();
    assert_eq!(graph.node_by_external_id("transient", None), None);
    assert_eq!(
        graph.allocated_entity_for_external_id("transient"),
        Some(ExternalEntityRef::Node(transient))
    );
    assert!(graph.upsert_nodes(vec![node("transient", 2)]).is_err());
    assert_eq!(graph.add_node(), transient + 1);
}

#[test]
fn edge_identity_changes_and_deletion_keep_history_and_old_names() {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(2);
    let id = graph
        .upsert_edges(vec![edge("old", nodes[0], nodes[1], 1)])
        .unwrap()[0];
    let first = graph.commit("initial".into(), "test".into()).unwrap();
    graph.set_edge_external_id(id, "new".into()).unwrap();
    let second = graph.commit("rename".into(), "test".into()).unwrap();
    assert_eq!(graph.edge_by_external_id("old", None), None);
    assert_eq!(graph.edge_by_external_id("new", None), Some(id));
    assert_eq!(
        graph
            .snapshot_at_commit(first)
            .unwrap()
            .edge_by_external_id("old", None),
        Some(id)
    );
    assert_eq!(
        graph
            .snapshot_at_commit(second)
            .unwrap()
            .edge_by_external_id("new", None),
        Some(id)
    );
    graph.remove_edge(id).unwrap();
    assert_eq!(graph.edge_by_external_id("new", None), None);
    assert_eq!(
        graph.allocated_entity_for_external_id("new"),
        Some(ExternalEntityRef::Edge(id))
    );
    assert!(graph
        .upsert_edges(vec![edge("old", nodes[0], nodes[1], 2)])
        .is_err());
    assert!(graph
        .upsert_edges(vec![edge("new", nodes[0], nodes[1], 2)])
        .is_err());
}
