//! Endpoint staging shares canonical identity, cascade and timing semantics.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        global_registry,
        pipeline::{Pipeline, PipelineBuilder},
        AlgorithmOutput, Context,
    },
    subgraphs::Subgraph,
    AttrValue, BatchNodeRef as N, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::sync::Arc;
fn fixture(directed: bool) -> Arc<OwnerCell<Graph>> {
    let mut g = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    g.apply_batch(
        GraphBatch {
            add_nodes: 4,
            add_edges: vec![
                (N::Added(0), N::Added(1)),
                (N::Added(0), N::Added(1)),
                (N::Added(1), N::Added(1)),
            ],
            edge_attrs: [(
                "w".into(),
                vec![(groggy::BatchEdgeRef::Added(0), AttrValue::Int(7))],
            )]
            .into(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "test".into(),
    )
    .unwrap();
    Arc::new(OwnerCell::new(g))
}
fn selected(g: &Arc<OwnerCell<Graph>>) -> Subgraph {
    Subgraph::from_frozen_view(g.clone(), g.borrow().read_view().unwrap(), "test".into())
}
fn pipeline(rows: Value) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":rows}).to_string());
        })
        .build(global_registry())
}
#[test]
fn endpoint_revision_rescues_edge_from_cascade_and_captures_real_new_adjacency() {
    for directed in [false, true] {
        let g = fixture(directed);
        let before = g.borrow().read_view().unwrap();
        let head = g.borrow().current_state_id();
        let mut ctx = Context::new();
        let result = pipeline(json!([
            {"id":"graph.capture_staged","params":{"output":"before"}},
            {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,3]],"effective_at":10}},
            {"id":"graph.remove_nodes","params":{"ids":[1],"effective_at":20}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.emit","params":{"source":"pending","name":"pending"}},
            {"id":"graph.emit","params":{"source":"before","name":"before"}}
        ]))
        .unwrap()
        .run(&mut ctx, selected(&g))
        .unwrap();
        assert_eq!(g.borrow().current_state_id(), head + 1);
        assert_eq!(result.edge_count(), 1);
        let output = ctx.take_outputs();
        let AlgorithmOutput::GraphView(view) = &output["pending"] else {
            panic!("graph")
        };
        assert_eq!(view.edge_ids(), vec![0]);
        assert_eq!(view.edge_endpoints(0).unwrap(), (2, 3));
        assert_eq!(view.neighbors(2).unwrap(), vec![3]);
        assert_eq!(
            view.neighbors(3).unwrap(),
            if directed { vec![] } else { vec![2] }
        );
        assert_eq!(
            view.edge_attr(0, &"w".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        assert_eq!(view.write_state_id(), None);
        assert_eq!(before.edge_endpoints(0).unwrap(), (0, 1));
        assert_eq!(g.borrow().edge_endpoints(0).unwrap(), (2, 3));
        use groggy::state::publication::ChangeKey;
        let meta = g.borrow().publication_metadata(head + 1).unwrap().unwrap();
        assert_eq!(meta.change_times[&ChangeKey::EdgeEndpoints(0)], 10);
        assert_eq!(meta.change_times[&ChangeKey::EdgeRemoved(1)], 20);
        assert_eq!(meta.change_times[&ChangeKey::EdgeRemoved(2)], 20);
    }
}
#[test]
fn duplicate_endpoint_removal_and_invalid_timeline_reject_the_entire_plan() {
    for suffix in [
        json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,3],[0,3,2]]}}]),
        json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,3]]}},{"id":"graph.remove_edges","params":{"ids":[0]}}]),
        json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,3]]}},{"id":"graph.remove_nodes","params":{"ids":[3]}}]),
        json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,3]],"effective_at":30}},{"id":"graph.remove_nodes","params":{"ids":[1],"effective_at":20}}]),
        json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,99,3]]}}]),
    ] {
        let g = fixture(false);
        let head = g.borrow().current_state_id();
        let mut ctx = Context::new();
        ctx.add_output("retained", AlgorithmOutput::Histogram(vec![9]));
        assert!(pipeline(suffix)
            .unwrap()
            .run(&mut ctx, selected(&g))
            .is_err());
        assert_eq!(g.borrow().current_state_id(), head);
        assert_eq!(g.borrow().edge_endpoints(0).unwrap(), (0, 1));
        assert_eq!(g.borrow().edge_count(), 3);
        assert!(
            matches!(&ctx.take_outputs()["retained"],AlgorithmOutput::Histogram(v) if v==&vec![9])
        );
    }
}
#[test]
fn malformed_controls_and_unselected_new_endpoints_reject() {
    for rows in [
        json!([[true, 0, 1]]),
        json!([[-1, 0, 1]]),
        json!([[0, 1]]),
        json!([[0, 1, 2, 3]]),
        json!([[0, 1.5, 2]]),
        json!([1, 2, 3]),
    ] {
        assert!(
            pipeline(json!([{"id":"graph.set_edge_endpoints","params":{"revisions":rows}}]))
                .is_err()
        );
    }
    let g = fixture(false);
    let head = g.borrow().current_state_id();
    let view = g.borrow().read_view().unwrap().select_edges(&[0]).unwrap();
    let scope = Subgraph::from_frozen_view(g.clone(), view, "edge".into());
    assert!(
        pipeline(json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,2]]}}]))
            .unwrap()
            .run(&mut Context::new(), scope)
            .is_err()
    );
    assert_eq!(g.borrow().current_state_id(), head);
    pipeline(json!([{"id":"graph.set_edge_endpoints","params":{"revisions":[]}}]))
        .unwrap()
        .run(&mut Context::new(), selected(&g))
        .unwrap();
    assert_eq!(g.borrow().current_state_id(), head);
}

#[test]
fn existing_edge_revisions_use_batch_global_new_node_refs_and_recover_exactly() {
    for directed in [false, true] {
        for removal_first in [false, true] {
            let g = fixture(directed);
            g.borrow_mut().remove_node(2).unwrap();
            let before = g.borrow().read_view().unwrap();
            let head = g.borrow().current_state_id();
            let mut edits = vec![
                json!({"id":"graph.add_nodes","params":{"count":1,"output":"first","effective_at":10}}),
                json!({"id":"graph.add_nodes","params":{"count":1,"output":"second","effective_at":15}}),
            ];
            let revision = json!({"id":"graph.set_edge_endpoints","params":{"revisions":[[0,{"added":0},{"added":1}]],"effective_at":20}});
            let removal = json!({"id":"graph.remove_nodes","params":{"ids":[1],"effective_at":30}});
            edits.extend(if removal_first {
                [removal, revision]
            } else {
                [revision, removal]
            });
            edits.extend([
                json!({"id":"graph.capture_staged","params":{"output":"pending"}}),
                json!({"id":"graph.emit","params":{"source":"pending","name":"pending"}}),
            ]);
            let mut ctx = Context::new();
            let result = pipeline(json!(edits))
                .unwrap()
                .run(&mut ctx, selected(&g))
                .unwrap();
            assert_eq!(result.nodes(), &[0, 3, 4, 5].into());
            assert_eq!(result.edges(), &[0].into());
            let AlgorithmOutput::GraphView(pending) = ctx.take_outputs().remove("pending").unwrap()
            else {
                panic!("missing capture");
            };
            assert_eq!(pending.edge_endpoints(0).unwrap(), (4, 5));
            assert_eq!(
                pending.edge_attr(0, &"w".into()).unwrap(),
                Some(AttrValue::Int(7))
            );
            assert_eq!(g.borrow().current_state_id(), head + 1);
            assert_eq!(before.edge_endpoints(0).unwrap(), (0, 1));
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("new-endpoints.gga");
            g.borrow().save_archive(&path).unwrap();
            let restored = Graph::load_archive(&path).unwrap();
            assert_eq!(restored.edge_endpoints(0).unwrap(), (4, 5));
            assert_eq!(
                restored
                    .read_view_at(19, None)
                    .unwrap()
                    .edge_endpoints(0)
                    .unwrap(),
                (0, 1)
            );
            assert_eq!(
                restored
                    .read_view_at(20, None)
                    .unwrap()
                    .edge_endpoints(0)
                    .unwrap(),
                (4, 5)
            );
            assert_eq!(
                restored
                    .publication_metadata(head + 1)
                    .unwrap()
                    .unwrap()
                    .change_times[&groggy::state::publication::ChangeKey::EdgeEndpoints(0)],
                20
            );
        }
    }
}

#[test]
fn new_endpoint_reference_failures_retain_outputs_and_consume_no_ids() {
    for (row, at) in [
        (json!([0, 0, {"added":1}]), 20),
        (json!([0, 0, {"added":0}]), 5),
        (json!([0, 0, 4]), 20),
    ] {
        let g = fixture(false);
        let head = g.borrow().current_state_id();
        let mut ctx = Context::new();
        ctx.add_output("retained", AlgorithmOutput::Histogram(vec![9]));
        let plan = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.set_edge_endpoints","params":{"revisions":[row],"effective_at":at}}
        ]))
        .unwrap();
        assert!(plan.run(&mut ctx, selected(&g)).is_err());
        assert_eq!(g.borrow().current_state_id(), head);
        assert_eq!(g.borrow().edge_endpoints(0).unwrap(), (0, 1));
        assert!(
            matches!(&ctx.take_outputs()["retained"], AlgorithmOutput::Histogram(v) if v == &vec![9])
        );
        assert_eq!(g.borrow_mut().add_node().unwrap(), 4);
    }
}
