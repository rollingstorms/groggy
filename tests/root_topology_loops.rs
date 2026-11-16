//! Sequential root loops share one pending topology plan and one acceptance.
use groggy::algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use serde_json::{json, Value};
use std::sync::Arc;

fn fixture() -> Arc<OwnerCell<Graph>> {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    Arc::new(OwnerCell::new(graph))
}
fn scope(graph: &Arc<OwnerCell<Graph>>) -> Subgraph {
    Subgraph::from_frozen_view(
        graph.clone(),
        graph.borrow().read_view().unwrap(),
        "root".into(),
    )
}
fn pipeline(steps: Value) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
}
fn body() -> Value {
    json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"output":"new_edges"}},
        {"id":"core.load_node_attr","params":{"attr":"counter","default":0,"target":"values"}},
        {"id":"core.init_scalar","params":{"value":1,"target":"one"}},
        {"id":"core.add","params":{"left":"values","right":"one","target":"next"}},
        {"id":"core.attach_node_attr","params":{"source":"next","attr":"counter"}},
        {"id":"graph.capture_staged","params":{"output":"pending"}},
        {"id":"graph.emit","params":{"source":"pending","name":"last"}}
    ])
}

#[test]
fn root_loops_refresh_selected_membership_and_keep_global_added_positions() {
    for nested in [false, true] {
        let owner = fixture();
        let head = owner.borrow().current_state_id();
        let input = scope(&owner);
        let before = owner.borrow().read_view().unwrap();
        let mut operation = json!({"id":"iter.loop","params":{"iterations":2,"body":body()}});
        let iterations = if nested { 4 } else { 2 };
        if nested {
            operation = json!({"id":"iter.loop","params":{"iterations":2,"body":[operation]}});
        }
        let pipe = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |p| {
                p.set_text(
                    "steps",
                    json!({"atomic":true,"steps":[operation]}).to_string(),
                )
            })
            .with_algorithm("centrality.pagerank", |p| p.set_text("output_attr", "rank"))
            .build(global_registry())
            .unwrap();
        let mut context = Context::new();
        let result = pipe.run_atomic_properties(&mut context, input).unwrap();
        assert_eq!(result.node_count(), 2 + iterations);
        let graph = owner.borrow();
        assert_eq!(graph.current_state_id(), head + 1);
        for node in 0..2 + iterations {
            let expected = if node < 2 {
                iterations
            } else {
                iterations + 2 - node
            };
            assert_eq!(
                graph
                    .get_node_attr(node, &"counter".into())
                    .unwrap()
                    .as_ref()
                    .and_then(AttrValue::as_int),
                Some(expected as i64)
            );
            assert!(graph.get_node_attr(node, &"rank".into()).unwrap().is_some());
        }
        for edge in 1..=iterations {
            assert_eq!(graph.edge_endpoints(edge).unwrap(), (0, 2));
        }
        assert_eq!(before.node_count(), 2);
        assert_eq!(before.node_attr(0, &"counter".into()).unwrap(), None);
        let outputs = context.take_outputs();
        let AlgorithmOutput::GraphView(last) = &outputs["last"] else {
            panic!("graph capture")
        };
        assert_eq!(last.node_count(), 2 + iterations);
        assert_eq!(last.node_attr(0, &"rank".into()).unwrap(), None);
    }
}

#[test]
fn repeated_removal_is_not_silently_ignored_and_failed_allocations_are_reusable() {
    let owner = fixture();
    let head = owner.borrow().current_state_id();
    let mut context = Context::new();
    context.add_output("prior", AlgorithmOutput::Histogram(vec![7]));
    let pipe = pipeline(json!([{"id":"iter.loop","params":{"iterations":2,"body":[
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"graph.remove_nodes","params":{"ids":[1]}}
    ]}}]))
    .unwrap();
    assert!(pipe.run(&mut context, scope(&owner)).is_err());
    assert_eq!(owner.borrow().current_state_id(), head);
    assert!(owner.borrow().has_node(1));
    assert_eq!(owner.borrow().edge_endpoints(0).unwrap(), (0, 1));
    assert!(
        matches!(&context.take_outputs()["prior"], AlgorithmOutput::Histogram(v) if v == &vec![7])
    );
    assert_eq!(owner.borrow_mut().add_node().unwrap(), 2);
}

#[test]
fn zero_iteration_root_topology_is_noop_but_fixed_child_topology_stays_guarded() {
    let owner = fixture();
    let head = owner.borrow().current_state_id();
    pipeline(json!([{"id":"iter.loop","params":{"iterations":0,"body":body()}}]))
        .unwrap()
        .run(&mut Context::new(), scope(&owner))
        .unwrap();
    assert_eq!(owner.borrow().current_state_id(), head);
    assert!(pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"sample.iterate_nodes","params":{"target":"seeds"}},
        {"id":"sample.for_each","params":{"source":"seeds","target":"out","body":[
            {"id":"iter.loop","params":{"iterations":0,"body":body()}}
        ]}}
    ]))
    .is_err());
    assert_eq!(owner.borrow().current_state_id(), head);
    assert_eq!(owner.borrow_mut().add_node().unwrap(), 2);
}

#[test]
fn mid_loop_cancellation_restores_outputs_ids_and_reusable_context() {
    use groggy::algorithms::steps::{
        global_step_registry, AtomicStepEffect, Step, StepMetadata, StepScope,
    };
    use groggy::algorithms::CostHint;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct CancelAfterTwo(Arc<AtomicUsize>);
    impl Step for CancelAfterTwo {
        fn id(&self) -> &'static str {
            "test.cancel_root_loop"
        }
        fn atomic_effect(&self) -> AtomicStepEffect {
            AtomicStepEffect::ImmutableInputs
        }
        fn apply(&self, ctx: &mut Context, _: &mut StepScope<'_>) -> anyhow::Result<()> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
                ctx.cancel_token().unwrap().store(true, Ordering::Relaxed);
            }
            Ok(())
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    global_step_registry()
        .register(
            "test.cancel_root_loop",
            StepMetadata {
                id: "test.cancel_root_loop".into(),
                description: "test cooperative cancellation".into(),
                cost_hint: CostHint::Linear,
            },
            move |_| Ok(Box::new(CancelAfterTwo(calls.clone()))),
        )
        .unwrap();
    let owner = fixture();
    let head = owner.borrow().current_state_id();
    let token = Arc::new(AtomicBool::new(false));
    let mut context = Context::with_cancel_token(token.clone());
    context.add_output("last", AlgorithmOutput::Histogram(vec![7]));
    let mut steps = body().as_array().unwrap().clone();
    steps.push(json!({"id":"test.cancel_root_loop","params":{}}));
    let error = pipeline(json!([{"id":"iter.loop","params":{"iterations":3,"body":steps}}]))
        .unwrap()
        .run(&mut context, scope(&owner))
        .unwrap_err();
    assert!(error.to_string().contains("cancel"));
    assert_eq!(owner.borrow().current_state_id(), head);
    assert_eq!(
        owner.borrow().get_node_attr(0, &"counter".into()).unwrap(),
        None
    );
    assert!(
        matches!(&context.take_outputs()["last"], AlgorithmOutput::Histogram(v) if v == &vec![7])
    );
    token.store(false, Ordering::Relaxed);
    pipeline(json!([{"id":"iter.loop","params":{"iterations":1,"body":body()}}]))
        .unwrap()
        .run(&mut context, scope(&owner))
        .unwrap();
    assert_eq!(owner.borrow().current_state_id(), head + 1);
    assert!(owner.borrow().has_node(2));
    assert!(!owner.borrow().has_node(3));
}
