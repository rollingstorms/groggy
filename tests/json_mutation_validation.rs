use groggy::{AttrValue as V, Graph};
#[test]
fn malformed_json_rejects_before_a_scalar_mutation() {
    for manual in [false, true] {
        let mut graph = if manual {
            Graph::new_manual()
        } else {
            Graph::new()
        };
        graph.add_node().unwrap();
        if manual {
            graph.commit("base".into(), "test".into()).unwrap();
        }
        let before = graph.current_state_id();
        assert!(graph
            .set_node_attr(0, "payload".into(), V::Json("{".into()))
            .is_err());
        assert_eq!(graph.current_state_id(), before);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.get_node_attr(0, &"payload".into()).unwrap(), None);
    }
}

use groggy::{BatchEdgeRef as E, BatchNodeRef as N, GraphBatch, NodeUpsert};
use std::collections::HashMap;
fn base(manual: bool) -> Graph {
    let mut graph = if manual {
        Graph::new_manual()
    } else {
        Graph::new()
    };
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    if manual {
        graph.commit("base".into(), "test".into()).unwrap();
    }
    graph
}
#[test]
fn malformed_later_bulk_values_and_creations_reject_without_prefixes_or_allocations() {
    for manual in [false, true] {
        let mut graph = base(manual);
        let state = graph.current_state_id();
        assert!(graph
            .set_node_attr_column(
                "payload".into(),
                vec![(0, V::Json("{}".into())), (1, V::Json("[] []".into()))]
            )
            .is_err());
        assert!(graph
            .set_edge_attr(0, "payload".into(), V::Json("[NaN]".into()))
            .is_err());
        assert!(graph
            .add_nodes_with_attrs(vec![
                HashMap::from([("payload".into(), V::Json("{}".into()))]),
                HashMap::from([("payload".into(), V::Json("".into()))])
            ])
            .is_err());
        assert!(graph
            .add_edges_with_attrs(vec![
                (0, 1, HashMap::new()),
                (
                    1,
                    0,
                    HashMap::from([("payload".into(), V::Json("{\"x\":}".into()))])
                )
            ])
            .is_err());
        assert!(graph
            .upsert_nodes(vec![NodeUpsert {
                external_id: "not-reserved".into(),
                attrs: HashMap::from([("payload".into(), V::Json("{".into()))])
            }])
            .is_err());
        assert_eq!(graph.current_state_id(), state);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(graph.pool().statistics().node_count, 2);
        assert_eq!(graph.pool().statistics().edge_count, 1);
        assert!(graph
            .pool()
            .get_node_attribute_column(&"payload".into())
            .is_none());
        assert!(graph
            .pool()
            .get_edge_attribute_column(&"payload".into())
            .is_none());
        assert_eq!(graph.allocated_entity_for_external_id("not-reserved"), None);
    }
}
#[test]
fn mixed_checked_batches_reject_bad_json_before_any_new_ids_or_valid_columns_are_written() {
    let mut graph = base(false);
    let before = graph.current_state_id();
    let batch = GraphBatch {
        add_nodes: 1,
        add_edges: vec![(N::Existing(0), N::Added(0))],
        node_attrs: HashMap::from([("valid".into(), vec![(N::Existing(0), V::Int(7))])]),
        edge_attrs: HashMap::from([(
            "payload".into(),
            vec![(E::Added(0), V::Json("{\"n\":01}".into()))],
        )]),
        node_external_ids: vec![(N::Added(0), "not-consumed".into())],
        ..Default::default()
    };
    assert!(graph
        .apply_batch(batch, before, "bad".into(), "test".into())
        .is_err());
    assert_eq!(graph.current_state_id(), before);
    assert_eq!(graph.pool().statistics().node_count, 2);
    assert_eq!(graph.pool().statistics().edge_count, 1);
    assert_eq!(graph.get_node_attr(0, &"valid".into()).unwrap(), None);
    assert_eq!(graph.allocated_entity_for_external_id("not-consumed"), None);
    assert!(graph.check_integrity().is_ok());
}
#[test]
fn valid_json_keeps_its_exact_text_and_large_numbers_through_history_archive_and_journal() {
    let text=" { \"big\": 1234567890123456789012345678901234567890, \"exponent\": 1e999, \"a\": [true, null] } \n";
    let mut graph = base(false);
    graph
        .set_node_attr(0, "payload".into(), V::Json(text.into()))
        .unwrap();
    graph
        .set_edge_attr(0, "payload".into(), V::Json("null".into()))
        .unwrap();
    graph
        .set_node_attr(1, "text".into(), V::Text("{".into()))
        .unwrap();
    let state = graph.current_state_id();
    let copy = graph.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(
        copy.get_node_attr(0, &"payload".into()).unwrap(),
        Some(V::Json(text.into()))
    );
    assert_eq!(
        copy.get_edge_attr(0, &"payload".into()).unwrap(),
        Some(V::Json("null".into()))
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal");
    let store = groggy::DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    drop(store);
    let store = groggy::DurableGraph::open(&path).unwrap();
    assert_eq!(store.graph().unwrap().current_state_id(), state);
    assert_eq!(
        store
            .graph()
            .unwrap()
            .get_node_attr(0, &"payload".into())
            .unwrap(),
        Some(V::Json(text.into()))
    );
    assert!(store.check_integrity().is_ok());
}
