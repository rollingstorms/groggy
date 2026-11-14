//! Terminal topology effects share one checked acceptance with property work.
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
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn fixture(directed: bool) -> Arc<OwnerCell<Graph>> {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new_publishing()
    };
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 5,
                add_edges: [(0, 1), (1, 2), (1, 2), (2, 2), (3, 4), (1, 4)]
                    .into_iter()
                    .map(|(s, t)| (N::Added(s), N::Added(t)))
                    .collect(),
                node_attrs: [("keep".into(), vec![(N::Added(3), AttrValue::Int(1))])].into(),
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    Arc::new(OwnerCell::new(graph))
}
fn scope(graph: &Arc<OwnerCell<Graph>>) -> Subgraph {
    let view = graph.borrow().read_view().unwrap();
    Subgraph::from_frozen_view(graph.clone(), view, "test".into())
}
fn pipeline(steps: Value, atomic: bool) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text(
                "steps",
                json!({"atomic": atomic,"steps": steps}).to_string(),
            );
        })
        .build(global_registry())
}
fn prefix() -> Vec<Value> {
    vec![
        json!({"id":"core.filter_nodes_by_attr","params":{"attr":"keep","predicate":{"type":"eq","value":{"type":"Int","value":1}},"target":"survivors"}}),
        json!({"id":"core.attach_node_attr","params":{"source":"survivors","attr":"marked"}}),
    ]
}

#[test]
fn calculations_and_multiple_removals_publish_once_with_cascade_clocks_and_retained_reads() {
    for directed in [false, true] {
        let graph = fixture(directed);
        let retained = graph.borrow().read_view().unwrap();
        let head = graph.borrow().current_state_id();
        let mut steps = prefix();
        steps.extend([
            json!({"id":"graph.remove_edges","params":{"ids":[4],"effective_at":30}}),
            json!({"id":"graph.remove_nodes","params":{"ids":[1],"effective_at":20}}),
        ]);
        let result = pipeline(json!(steps), true)
            .unwrap()
            .run(&mut Context::new(), scope(&graph))
            .unwrap();
        let owner = graph.borrow();
        assert_eq!(owner.current_state_id(), head + 1);
        assert!(!owner.has_uncommitted_changes());
        let metadata = owner.publication_metadata(head + 1).unwrap().unwrap();
        use groggy::state::publication::ChangeKey;
        assert_eq!(metadata.change_times[&ChangeKey::NodeRemoved(1)], 20);
        assert_eq!(metadata.change_times[&ChangeKey::EdgeRemoved(4)], 30);
        for id in [0, 1, 2, 5] {
            assert_eq!(metadata.change_times[&ChangeKey::EdgeRemoved(id)], 20);
        }
        assert_eq!(
            metadata.change_times[&ChangeKey::NodeAttribute(3, "marked".into())],
            metadata.recorded_at
        );
        assert_eq!(result.nodes().len(), 4);
        assert_eq!(result.edges().iter().copied().collect::<Vec<_>>(), vec![3]);
        assert_eq!(
            owner.get_node_attr(3, &"marked".into()).unwrap(),
            Some(AttrValue::Int(1))
        );
        assert_eq!(retained.node_count(), 5);
        assert_eq!(retained.edge_count(), 6);
        assert!(retained.node_attr(3, &"marked".into()).unwrap().is_none());
    }
}

#[test]
fn late_target_duplicate_and_property_conflicts_leave_graph_and_outputs_unchanged() {
    for suffix in [
        json!([{"id":"graph.remove_nodes","params":{"ids":[1]}},{"id":"graph.remove_edges","params":{"ids":[999]}}]),
        json!([{"id":"graph.remove_nodes","params":{"ids":[1]}},{"id":"graph.remove_nodes","params":{"ids":[1]}}]),
        json!([{"id":"graph.remove_nodes","params":{"ids":[3]}}]),
    ] {
        let graph = fixture(false);
        let head = graph.borrow().current_state_id();
        let mut ctx = Context::new();
        ctx.add_output("prior", AlgorithmOutput::Histogram(vec![3; 100]));
        let mut steps = prefix();
        steps.extend(suffix.as_array().unwrap().iter().cloned());
        assert!(pipeline(json!(steps), true)
            .unwrap()
            .run(&mut ctx, scope(&graph))
            .is_err());
        assert_eq!(graph.borrow().current_state_id(), head);
        assert_eq!(graph.borrow().edge_count(), 6);
        assert_eq!(
            graph.borrow().get_node_attr(3, &"marked".into()).unwrap(),
            None
        );
        assert!(ctx.take_outputs().contains_key("prior"));
        pipeline(
            json!([{"id":"graph.remove_nodes","params":{"ids":[]}}]),
            true,
        )
        .unwrap()
        .run(&mut ctx, scope(&graph))
        .unwrap();
        assert_eq!(graph.borrow().current_state_id(), head);
    }
}

#[test]
fn map_keys_and_exact_scope_are_validated_before_acceptance() {
    let graph = fixture(false);
    let steps = json!([
        {"id":"core.filter_nodes_by_attr","params":{"attr":"keep","predicate":{"type":"eq","value":{"type":"Int","value":1}},"target":"chosen"}},
        {"id":"graph.remove_nodes","params":{"source":"chosen"}}
    ]);
    pipeline(steps, true)
        .unwrap()
        .run(&mut Context::new(), scope(&graph))
        .unwrap();
    assert!(!graph.borrow().has_node(3));
    assert!(!graph.borrow().has_edge(4));
    let selected = graph
        .borrow()
        .read_view()
        .unwrap()
        .select_nodes(&[0])
        .unwrap();
    let subgraph = Subgraph::from_frozen_view(graph.clone(), selected, "narrow".into());
    let head = graph.borrow().current_state_id();
    assert!(pipeline(
        json!([{"id":"graph.remove_nodes","params":{"ids":[2]}}]),
        true
    )
    .unwrap()
    .run(&mut Context::new(), subgraph)
    .is_err());
    assert_eq!(graph.borrow().current_state_id(), head);
}

#[test]
fn unsafe_order_controls_and_outer_property_scopes_reject_before_execution() {
    let remove = json!({"id":"graph.remove_nodes","params":{"ids":[1]}});
    assert!(pipeline(json!([remove.clone()]), false).is_err());
    assert!(pipeline(
        json!([remove.clone(),{"id":"core.init_nodes","params":{"target":"late","value":1}}]),
        true
    )
    .is_err());
    for params in [
        json!({"ids":[true]}),
        json!({"ids":[-1]}),
        json!({"ids":[1],"source":"x"}),
        json!({"ids":[],"effective_at":1.5}),
        json!({"ids":[],"unknown":true}),
    ] {
        assert!(pipeline(json!([{"id":"graph.remove_nodes","params":params}]), true).is_err());
    }
    let graph = fixture(false);
    let head = graph.borrow().current_state_id();
    let pipe = pipeline(json!([remove]), true).unwrap();
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    assert!(pipe.run(&mut ctx, scope(&graph)).is_err());
    ctx.set_persist_results(true);
    assert!(pipe.run_atomic_properties(&mut ctx, scope(&graph)).is_err());
    assert_eq!(graph.borrow().current_state_id(), head);
    let cancelled = Arc::new(AtomicBool::new(true));
    assert!(pipe
        .run(&mut Context::with_cancel_token(cancelled), scope(&graph))
        .is_err());
    assert_eq!(graph.borrow().current_state_id(), head);
}

#[test]
fn successful_exact_selection_does_not_expand_omitted_parallel_edges() {
    let graph = fixture(false);
    let view = graph
        .borrow()
        .read_view()
        .unwrap()
        .select_edges(&[1, 4])
        .unwrap();
    let selected = Subgraph::from_frozen_view(graph.clone(), view, "exact".into());
    let result = pipeline(
        json!([{"id":"graph.remove_edges","params":{"ids":[4]}}]),
        true,
    )
    .unwrap()
    .run(&mut Context::new(), selected)
    .unwrap();
    assert_eq!(result.edges().iter().copied().collect::<Vec<_>>(), vec![1]);
    assert_eq!(result.node_count(), 4);
    assert!(graph.borrow().has_edge(2));
}
