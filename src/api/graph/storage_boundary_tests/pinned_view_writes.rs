//! Scoped mutation uses a frozen source pin, prevalidated output and one acceptance.
use crate::{
    AttrValue as V, BatchChangeRef as T, BatchEdgeRef as E, BatchNodeRef as N, ChangeKey as K,
    DurableGraph, Graph, GraphBatch, GraphConfig, GraphSchema, GraphView,
};
use std::collections::{HashMap, HashSet};

fn fixture(directed: bool, publishing: bool) -> Graph {
    let mut graph = if directed {
        Graph::new_directed_manual()
    } else {
        Graph::new_manual()
    };
    graph.set_automatic_publication(publishing).unwrap();
    graph.insert_nodes(5).unwrap();
    graph
        .add_edges(&[(0, 1), (0, 1), (1, 1), (2, 3), (3, 4)])
        .unwrap();
    graph.set_node_attr(0, "value".into(), V::Int(1)).unwrap();
    graph.commit("fixture".into(), "test".into()).unwrap();
    graph
}
fn score(id: usize, value: i64) -> GraphBatch {
    GraphBatch {
        node_attrs: HashMap::from([("value".into(), vec![(N::Existing(id), V::Int(value))])]),
        ..Default::default()
    }
}
fn assert_current_pin(graph: &Graph, view: &GraphView) {
    let live = graph.read_view().unwrap();
    let a = live.read_state().snapshot();
    let b = view.read_state().snapshot();
    assert_eq!(a.state_id, b.state_id);
    assert_eq!(a.active_nodes, b.active_nodes);
    assert_eq!(a.edges, b.edges);
    assert_eq!(a.node_attributes, b.node_attributes);
    assert_eq!(a.edge_attributes, b.edge_attributes);
    assert_eq!(view.write_state_id(), Some(graph.current_state_id()));
}

#[test]
fn mixed_writes_refresh_only_selected_membership_and_can_chain() {
    for directed in [false, true] {
        for publishing in [false, true] {
            // Timed topology needs a precisely timed birth, independent of the
            // ordinary/manual policy exercised by the scoped write below.
            let mut graph = if directed {
                Graph::new_directed_manual()
            } else {
                Graph::new_manual()
            };
            graph
                .apply_batch(
                    GraphBatch {
                        add_nodes: 5,
                        add_edges: vec![
                            (N::Added(0), N::Added(1)),
                            (N::Added(0), N::Added(1)),
                            (N::Added(1), N::Added(1)),
                            (N::Added(2), N::Added(3)),
                            (N::Added(3), N::Added(4)),
                        ],
                        node_attrs: HashMap::from([(
                            "value".into(),
                            vec![(N::Added(0), V::Int(1))],
                        )]),
                        effective_at: Some(0),
                        ..Default::default()
                    },
                    0,
                    "timed fixture".into(),
                    "test".into(),
                )
                .unwrap();
            graph.set_automatic_publication(publishing).unwrap();
            let view = graph.read_view().unwrap().select_edges(&[0, 2]).unwrap();
            let head = graph.current_state_id();
            let batch = GraphBatch {
                add_nodes: 1,
                add_edges: vec![(N::Existing(1), N::Added(0))],
                node_attrs: HashMap::from([
                    (
                        "value".into(),
                        vec![
                            (N::Existing(0), V::Float64(1.0000000000000002)),
                            (N::Added(0), V::Null),
                        ],
                    ),
                    (
                        "labels".into(),
                        vec![(N::Added(0), V::TextVec(vec!["z".into(), "a".into()]))],
                    ),
                ]),
                edge_attrs: HashMap::from([(
                    "weight".into(),
                    vec![(E::Existing(0), V::Float64(7.5)), (E::Added(0), V::Int(2))],
                )]),
                node_external_ids: vec![(N::Added(0), "new".into())],
                remove_edges: vec![E::Existing(2)],
                effective_at: Some(20),
                change_times: vec![(T::NodeAttribute(N::Existing(0), "value".into()), -10)],
                ..Default::default()
            };
            let result = graph
                .apply_view_batch(&view, batch, "mixed".into(), "test".into())
                .unwrap();
            assert_eq!(result.batch.state_id, head + 1);
            assert_eq!(result.batch.added_nodes, vec![5]);
            assert_eq!(result.batch.added_edges, vec![5]);
            assert_eq!(result.view.node_ids(), vec![0, 1, 5]);
            assert_eq!(result.view.edge_ids(), vec![0, 5]);
            assert!(!result.view.contains_edge(1)); // excluded parallel instance stays excluded
            assert!(graph.has_edge(1));
            assert_eq!(view.node_attr(0, &"value".into()).unwrap(), Some(V::Int(1)));
            assert!(view.contains_edge(2));
            assert_eq!(
                result.view.node_attr(5, &"labels".into()).unwrap(),
                Some(V::TextVec(vec!["a".into(), "z".into()]))
            );
            assert_eq!(graph.node_by_external_id("new", None), Some(5));
            assert_eq!(
                graph
                    .publication_metadata(head + 1)
                    .unwrap()
                    .unwrap()
                    .change_times[&K::NodeAttribute(0, "value".into())],
                -10
            );
            assert_eq!(graph.automatic_publication(), publishing);
            assert!(!graph.has_uncommitted_changes());
            assert_current_pin(&graph, &result.view);
            let next = graph
                .apply_view_batch(&result.view, score(5, 9), "next".into(), "test".into())
                .unwrap();
            assert_eq!(next.batch.state_id, head + 2);
            assert_eq!(
                result.view.node_attr(5, &"value".into()).unwrap(),
                Some(V::Null)
            );
            assert_current_pin(&graph, &next.view);
            assert!(graph
                .apply_view_batch(&view, score(0, 4), "stale".into(), "test".into())
                .is_err());
            assert_eq!(graph.current_state_id(), head + 2);
        }
    }
}

#[test]
fn cascades_require_every_affected_edge_in_scope() {
    let mut graph = fixture(true, true);
    let partial = graph.read_view().unwrap().select_edges(&[0]).unwrap();
    let before = graph.current_state_id();
    let bad = GraphBatch {
        add_nodes: 1,
        remove_nodes: vec![N::Existing(0)],
        ..Default::default()
    };
    let error = graph
        .apply_view_batch(&partial, bad, "scope".into(), "test".into())
        .unwrap_err()
        .to_string();
    assert!(error.contains("cascade to edge 1 outside"), "{error}");
    assert_eq!(graph.current_state_id(), before);
    assert_eq!(graph.node_count(), 5);
    let complete = graph.read_view().unwrap().select_nodes(&[0, 1]).unwrap();
    let result = graph
        .apply_view_batch(
            &complete,
            GraphBatch {
                remove_nodes: vec![N::Existing(0)],
                ..Default::default()
            },
            "delete".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(result.view.node_ids(), vec![1]);
    assert_eq!(result.view.edge_ids(), vec![2]);
    assert_eq!(graph.edge_ids_set(), HashSet::from([2, 3, 4]));
    assert!(complete.contains_node(0));
    assert_current_pin(&graph, &result.view);
    assert_eq!(graph.insert_nodes(1).unwrap(), vec![5]); // rejected allocation not consumed
}

#[test]
fn all_existing_mutation_targets_and_edge_endpoints_must_be_selected() {
    for bad in [
        score(2, 7),
        GraphBatch {
            edge_attrs: HashMap::from([("w".into(), vec![(E::Existing(1), V::Int(1))])]),
            ..Default::default()
        },
        GraphBatch {
            remove_edges: vec![E::Existing(1)],
            ..Default::default()
        },
        GraphBatch {
            remove_nodes: vec![N::Existing(2)],
            ..Default::default()
        },
        GraphBatch {
            add_edges: vec![(N::Existing(0), N::Existing(2))],
            ..Default::default()
        },
        GraphBatch {
            node_external_ids: vec![(N::Existing(2), "outside".into())],
            ..Default::default()
        },
        GraphBatch {
            edge_external_ids: vec![(E::Existing(1), "outside".into())],
            ..Default::default()
        },
    ] {
        let mut graph = fixture(false, true);
        let view = graph.read_view().unwrap().select_edges(&[0]).unwrap();
        let head = graph.current_state_id();
        assert!(graph
            .apply_view_batch(&view, bad, "bad".into(), "test".into())
            .unwrap_err()
            .to_string()
            .contains("outside the view"));
        assert_eq!(graph.current_state_id(), head);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.node_count(), 5);
        assert!(graph.allocated_entity_for_external_id("outside").is_none());
    }
}

#[test]
fn foreign_branch_dirty_and_historical_pins_fail_without_mutation() {
    let mut graph = fixture(false, true);
    let view = graph.read_view().unwrap();
    let mut foreign = fixture(false, true);
    assert!(foreign
        .apply_view_batch(&view, score(0, 2), "bad".into(), "test".into())
        .unwrap_err()
        .to_string()
        .contains("different graph"));
    graph.create_branch("sibling".into()).unwrap();
    graph.checkout_branch("sibling".into()).unwrap();
    assert!(graph
        .apply_view_batch(&view, score(0, 2), "bad".into(), "test".into())
        .unwrap_err()
        .to_string()
        .contains("different branch"));
    graph.checkout_branch("main".into()).unwrap();
    let head = graph.current_state_id();
    let historical = GraphView::all(graph.snapshot_at_commit(head).unwrap().read_state().clone());
    assert_eq!(historical.write_state_id(), None);
    assert!(graph
        .apply_view_batch(&historical, score(0, 2), "bad".into(), "test".into())
        .is_err());
    graph.set_automatic_publication(false).unwrap();
    graph.add_node().unwrap();
    let dirty = graph.read_view().unwrap();
    assert_eq!(dirty.write_state_id(), None);
    assert!(graph
        .apply_view_batch(&view, score(0, 2), "bad".into(), "test".into())
        .is_err());
    graph.commit("manual".into(), "test".into()).unwrap();
    graph.set_automatic_publication(true).unwrap();
    assert!(graph
        .apply_view_batch(&dirty, score(0, 2), "bad".into(), "test".into())
        .unwrap_err()
        .to_string()
        .contains("read-only"));
}

#[test]
fn raw_untracked_changes_and_corrupt_indices_cannot_be_overwritten_from_a_pin() {
    for corrupt in [false, true] {
        let mut graph = fixture(false, true);
        let view = graph.read_view().unwrap();
        let head = graph.current_state_id();
        let index = if corrupt {
            999
        } else {
            graph.pool_mut().set_attr("value".into(), V::Int(99), true)
        };
        graph
            .space_mut()
            .set_node_attr_index(0, "value".into(), index);
        assert!(graph
            .apply_view_batch(&view, score(0, 2), "bad".into(), "test".into())
            .is_err());
        assert_eq!(graph.current_state_id(), head);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(view.node_attr(0, &"value".into()).unwrap(), Some(V::Int(1)));
    }
}

#[test]
fn schema_and_batch_validation_preserve_head_and_allocation() {
    let mut graph = fixture(false, true);
    let view = graph.read_view().unwrap();
    let schema =
        GraphSchema::from_json(r#"{"node_labels":{"Person":{"name":{"type":"text"}}}}"#).unwrap();
    let bad = GraphBatch {
        add_nodes: 1,
        node_attrs: HashMap::from([(
            "labels".into(),
            vec![(N::Added(0), V::TextVec(vec!["Person".into()]))],
        )]),
        ..Default::default()
    };
    assert!(graph
        .apply_view_batch_with_schema(&view, bad, "bad".into(), "test".into(), &schema)
        .is_err());
    let duplicate = GraphBatch {
        node_attrs: HashMap::from([(
            "v".into(),
            vec![(N::Existing(0), V::Int(1)), (N::Existing(0), V::Int(2))],
        )]),
        ..Default::default()
    };
    assert!(graph
        .apply_view_batch(&view, duplicate, "bad".into(), "test".into())
        .is_err());
    assert!(graph
        .apply_view_batch(&view, GraphBatch::default(), "empty".into(), "test".into())
        .is_err());
    assert_eq!(graph.current_state_id(), view.write_state_id().unwrap());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.insert_nodes(1).unwrap(), vec![5]);
}

#[test]
fn empty_selections_accept_new_members_and_transient_lifecycles_stay_absent() {
    let mut graph = fixture(false, true);
    let view = graph.read_view().unwrap().select_nodes(&[]).unwrap();
    let result = graph
        .apply_view_batch(
            &view,
            GraphBatch {
                add_nodes: 2,
                remove_nodes: vec![N::Added(0)],
                add_edges: vec![(N::Added(1), N::Added(1))],
                remove_edges: vec![E::Added(0)],
                ..Default::default()
            },
            "lifecycle".into(),
            "test".into(),
        )
        .unwrap();
    assert_eq!(result.batch.added_nodes, vec![5, 6]);
    assert_eq!(result.view.node_ids(), vec![6]);
    assert!(result.view.edge_ids().is_empty());
    assert_current_pin(&graph, &result.view);
}

#[test]
fn projected_output_matches_actual_published_state_across_long_update_chains() {
    for directed in [false, true] {
        let mut graph = fixture(directed, true);
        let mut view = graph.read_view().unwrap();
        for step in 0..40 {
            let mut batch = score(0, step);
            if step % 2 == 0 {
                batch.add_edges = vec![(N::Existing(0), N::Existing(0))];
            } else {
                batch.remove_edges = vec![E::Existing(graph.edge_ids().into_iter().max().unwrap())];
            }
            let result = graph
                .apply_view_batch(&view, batch, "chain".into(), "test".into())
                .unwrap();
            assert_current_pin(&graph, &result.view);
            view = result.view;
        }
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<GraphView>();
}

#[test]
fn durable_view_batches_sync_recover_and_reject_without_appending() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("view.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                ..Default::default()
            },
            0,
            "init".into(),
            "test".into(),
        )
        .unwrap();
    let view = store.read_view().unwrap().select_nodes(&[0]).unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(store
        .apply_view_batch(&view, score(1, 3), "outside".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
    let result = store
        .apply_view_batch(&view, score(0, 7), "accepted".into(), "test".into())
        .unwrap();
    assert_eq!(result.batch.state_id, 2);
    assert_eq!(view.node_attr(0, &"value".into()).unwrap(), None);
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(store
        .apply_view_batch(&view, score(0, 8), "stale".into(), "test".into())
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
    drop(store);
    let mut store = DurableGraph::open(&path).unwrap();
    assert_eq!(
        store
            .graph()
            .unwrap()
            .get_node_attr(0, &"value".into())
            .unwrap(),
        Some(V::Int(7))
    );
    assert!(store
        .apply_view_batch(&result.view, score(0, 8), "old owner".into(), "test".into())
        .is_err());
    let fresh = store.read_view().unwrap();
    assert_eq!(
        store
            .apply_view_batch(&fresh, score(0, 9), "fresh".into(), "test".into())
            .unwrap()
            .batch
            .state_id,
        3
    );
}

#[test]
fn unrecorded_snapshot_materialization_requires_publication_before_view_writes() {
    let graph = fixture(false, true);
    let snapshot = graph.read_view().unwrap().read_state().snapshot().clone();
    let mut materialized = Graph::from_snapshot(snapshot).unwrap();
    let pin = materialized.read_view().unwrap();
    assert_eq!(pin.write_state_id(), None);
    assert!(materialized
        .apply_view_batch(&pin, score(0, 5), "unrecorded".into(), "test".into())
        .is_err());
    assert_eq!(
        materialized.get_node_attr(0, &"value".into()).unwrap(),
        Some(V::Int(1))
    );
}
