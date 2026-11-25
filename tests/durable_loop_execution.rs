//! Ordinary loop bodies prepare sealed children once, before leaving the owner.
use groggy::{
    algorithms::{
        global_registry,
        steps::{
            global_step_registry, read_scope::ReadStep, AtomicStepEffect, InitNodesStep, Step,
            StepMetadata, StepScope,
        },
        AlgorithmOutput, AlgorithmParamValue, Context, CostHint, Pipeline, PipelineBuilder,
    },
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{fs, thread::ThreadId};

fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn repeat(iterations: usize, body: Value) -> Value {
    json!({"id":"iter.loop","params":{"iterations":iterations,"body":body}})
}
fn fixture() -> Graph {
    let mut graph = Graph::new_directed();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 3,
                add_edges: vec![(N::Added(0), N::Added(1))],
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    graph
}
fn nested_program() -> Value {
    json!([
        {"id":"core.init_nodes","params":{"target":"zero","value":0}},
        {"id":"core.attach_node_attr","params":{"source":"zero","attr":"x","effective_at":10}},
        repeat(2, json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"x":0},"effective_at":10}},
            repeat(3, json!([
                {"id":"graph.capture_staged","params":{"output":"before"}},
                {"id":"core.load_node_attr","params":{"target":"x","attr":"x"}},
                {"id":"core.init_nodes","params":{"target":"one","value":1}},
                {"id":"core.add","params":{"left":"x","right":"one","target":"next"}},
                {"id":"core.attach_node_attr","params":{"source":"next","attr":"x","effective_at":20}}
            ]))
        ])),
        {"id":"graph.emit","params":{"source":"before","name":"before"}}
    ])
}

#[test]
fn nested_bodies_run_on_worker_read_pending_writes_and_publish_once() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("loop.journal");
    let graph = fixture();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    let selected = store.read_view().unwrap().select_nodes(&[0, 1]).unwrap();
    let head = graph.current_state_id();
    let bytes = fs::read(&path).unwrap();
    let p = pipeline(nested_program());
    let plan = p.prepare_durable(&store, &selected).unwrap();
    let mut ctx = Context::new();
    let prepared = std::thread::scope(|threads| {
        threads
            .spawn(|| plan.execute(&mut ctx).unwrap())
            .join()
            .unwrap()
    });
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let accepted = prepared.accept(&mut store).unwrap();
    assert_eq!(accepted.node_ids(), vec![0, 1, 3, 4]);
    assert_eq!(accepted.write_state_id(), Some(head + 1));
    for (id, value) in [(0, 6), (1, 6), (3, 6), (4, 3)] {
        assert_eq!(
            accepted.node_attr(id, &"x".into()).unwrap(),
            Some(V::Int(value))
        );
    }
    assert_eq!(
        store
            .read_view()
            .unwrap()
            .node_attr(2, &"x".into())
            .unwrap(),
        None
    );
    let outputs = ctx.take_outputs();
    let AlgorithmOutput::GraphView(before) = &outputs["before"] else {
        panic!("view")
    };
    assert_eq!(before.node_attr(0, &"x".into()).unwrap(), Some(V::Int(5)));
    assert_eq!(before.node_attr(4, &"x".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(before.write_state_id(), None);
    assert_eq!(selected.node_ids(), vec![0, 1]);
    assert_eq!(selected.node_attr(0, &"x".into()).unwrap(), None);
    let owner = graph.into_shared();
    let memory = Subgraph::from_frozen_view(
        owner.clone(),
        owner
            .borrow()
            .read_view()
            .unwrap()
            .select_nodes(&[0, 1])
            .unwrap(),
        "test".into(),
    );
    p.run(&mut Context::new(), memory).unwrap();
    for id in accepted.node_ids() {
        assert_eq!(
            owner.borrow().get_node_attr(id, &"x".into()).unwrap(),
            accepted.node_attr(id, &"x".into()).unwrap()
        );
    }
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(reopened.graph().unwrap().current_state_id(), head + 1);
    assert_eq!(
        reopened
            .read_view_at(19, None)
            .unwrap()
            .node_attr(0, &"x".into())
            .unwrap(),
        None
    );
    assert_eq!(
        reopened
            .read_view_at(20, None)
            .unwrap()
            .node_attr(0, &"x".into())
            .unwrap(),
        Some(V::Int(6))
    );
}

#[test]
fn inner_failure_discards_staged_allocations_outputs_and_journal() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("failure.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let head = selected.write_state_id();
    let p = pipeline(json!([repeat(
        2,
        json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.emit","params":{"source":"pending","name":"discard"}},
            repeat(1, json!([
                {"id":"graph.algorithm_table","params":{"source":"pending","algorithm":"bfs","start":999,"output":"bad"}}
            ]))
        ])
    )]));
    let mut ctx = Context::new();
    ctx.add_output("retained", AlgorithmOutput::GraphView(selected.clone()));
    let error = p
        .run_durable(&mut ctx, &mut store, &selected)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Loop iteration 1/2"), "{error}");
    assert!(error.contains("Loop iteration 1/1"), "{error}");
    assert_eq!(
        ctx.take_outputs().keys().collect::<Vec<_>>(),
        vec!["retained"]
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(store.read_view().unwrap().write_state_id(), head);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 3);
}

#[test]
fn unsupported_nested_body_is_rejected_even_at_zero_iterations() {
    for iterations in [0, 1] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("unsupported.journal");
        let store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            repeat(iterations, json!([repeat(0, json!([
                {"id":"core.weighted_degree","params":{"target":"bad","weight_attr":"weight"}}
            ]))]))
        ]));
        assert!(p.prepare_durable(&store, &selected).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}

#[test]
fn zero_iteration_bindings_are_sequential_independent_and_do_not_publish() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bindings.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let body = json!({"id":"iter.loop","params":{"iterations":0,"body":[],"loop_vars":[["source","alias"],["alias","again"]]}});
    let mut ctx = Context::new();
    pipeline(json!([
        {"id":"graph.capture_staged","params":{"output":"source"}},
        body,
        {"id":"graph.emit","params":{"source":"again","name":"alias"}}
    ]))
    .run_durable(&mut ctx, &mut store, &selected)
    .unwrap();
    let outputs = ctx.take_outputs();
    let AlgorithmOutput::GraphView(alias) = &outputs["alias"] else {
        panic!("view")
    };
    assert_eq!(alias.node_ids(), selected.node_ids());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let bad = pipeline(
        json!([{"id":"iter.loop","params":{"iterations":0,"body":[],"loop_vars":[["missing","alias"]]}}]),
    );
    assert!(bad
        .run_durable(&mut ctx, &mut store, &selected)
        .unwrap_err()
        .to_string()
        .contains("'missing' not found"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

// The wrapper deliberately guards every Step callback, while exposing only an
// existing sealed native kernel. An effect declaration alone cannot do this.
struct OwnerOnlyStep {
    owner: ThreadId,
    native: InitNodesStep,
}
impl OwnerOnlyStep {
    fn check_owner(&self) {
        assert_eq!(std::thread::current().id(), self.owner);
    }
}
impl Step for OwnerOnlyStep {
    fn id(&self) -> &'static str {
        self.check_owner();
        "test.owner_only_loop_leaf"
    }
    fn metadata(&self) -> StepMetadata {
        self.check_owner();
        StepMetadata {
            id: self.id().into(),
            description: "owner guard".into(),
            cost_hint: CostHint::Linear,
        }
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        self.check_owner();
        AtomicStepEffect::PendingProperties
    }
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        self.check_owner();
        self.native.read_kernel()
    }
    fn apply(&self, _: &mut Context, _: &mut StepScope<'_>) -> anyhow::Result<()> {
        panic!("prepared execution must invoke the sealed kernel")
    }
}
#[test]
fn worker_never_calls_step_metadata_factories_or_owner_apply() {
    let owner = std::thread::current().id();
    // Initialize core registration before registering the unique guarded leaf.
    let _ = pipeline(json!([]));
    global_step_registry()
        .register(
            "test.owner_only_loop_leaf",
            StepMetadata {
                id: "test.owner_only_loop_leaf".into(),
                description: "owner guard".into(),
                cost_hint: CostHint::Linear,
            },
            move |_| {
                assert_eq!(std::thread::current().id(), owner);
                Ok(Box::new(OwnerOnlyStep {
                    owner,
                    native: InitNodesStep::new("answer", AlgorithmParamValue::Int(17)),
                }))
            },
        )
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("owner.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let p = pipeline(json!([
        repeat(2, json!([repeat(3, json!([{"id":"test.owner_only_loop_leaf","params":{}}]))])),
        {"id":"core.attach_node_attr","params":{"source":"answer","attr":"answer","effective_at":10}}
    ]));
    let plan = p.prepare_durable(&store, &selected).unwrap();
    let prepared = std::thread::scope(|threads| {
        threads
            .spawn(move || plan.execute(&mut Context::new()).unwrap())
            .join()
            .unwrap()
    });
    let accepted = prepared.accept(&mut store).unwrap();
    assert_eq!(
        accepted.node_attr(0, &"answer".into()).unwrap(),
        Some(V::Int(17))
    );
}

#[test]
fn ordinary_body_can_nest_lowered_register_plan_without_publishing_inner_versions() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("mixed.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let head = selected.write_state_id().unwrap();
    let p = pipeline(json!([
        {"id":"core.init_nodes","params":{"target":"seed","value":2}},
        {"id":"core.attach_node_attr","params":{"source":"seed","attr":"x","effective_at":10}},
        repeat(3, json!([
            {"id":"iter.loop","params":{"iterations":1,"body":[],"batch_plan":{
                "slot_count":3,"carried_slots":[],"instructions":[
                    {"type":"load_node_attr","dst":0,"attr_name":"x","default":0.},
                    {"type":"load_scalar","dst":1,"value":2.},
                    {"type":"mul","dst":2,"lhs":0,"rhs":1},
                    {"type":"store_node_prop","src":2,"var_name":"answer"}
                ]
            }}},
            {"id":"core.attach_node_attr","params":{"source":"answer","attr":"x","effective_at":20}}
        ]))
    ]));
    let token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let bytes = fs::read(&path).unwrap();
    let mut cancelled = Context::with_cancel_token(token);
    assert!(p
        .prepare_durable(&store, &selected)
        .unwrap()
        .execute(&mut cancelled)
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let accepted = p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .unwrap();
    assert_eq!(
        accepted.node_attr(0, &"x".into()).unwrap(),
        Some(V::Float64(16.))
    );
    assert_eq!(accepted.write_state_id(), Some(head + 1));
}
