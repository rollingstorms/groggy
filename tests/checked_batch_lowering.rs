//! The serialized numeric plan is a frontend to the same checked native kernels.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{
    fs,
    sync::{atomic::AtomicBool, Arc},
};
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn plan(instructions: Value, slots: usize, carried: Value) -> Value {
    json!({"instructions":instructions,"slot_count":slots,"carried_slots":carried})
}
fn loop_step(plan: Value, iterations: usize) -> Value {
    json!({"id":"iter.loop","params":{"body":[],"batch_plan":plan,"iterations":iterations}})
}
fn fixture(directed: bool) -> Graph {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 3,
                add_edges: vec![
                    (N::Added(0), N::Added(2)),
                    (N::Added(0), N::Added(2)),
                    (N::Added(2), N::Added(2)),
                ],
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    graph
        .apply_batch(
            GraphBatch {
                remove_nodes: vec![N::Existing(1)],
                effective_at: Some(0),
                ..Default::default()
            },
            graph.current_state_id(),
            "sparse".into(),
            "test".into(),
        )
        .unwrap();
    graph
}
fn cases() -> Vec<(Value, Value)> {
    let mut cases = Vec::new();
    for name in ["add", "sub", "mul", "div", "min", "max"] {
        cases.push((json!({"type":name,"dst":2,"lhs":0,"rhs":1}),json!({"id":format!("core.{name}"),"params":{"left":"a","right":"b","target":"answer"}})));
    }
    cases.push((
        json!({"type":"pow","dst":2,"base":0,"exp":1}),
        json!({"id":"core.pow","params":{"left":"a","right":"b","target":"answer"}}),
    ));
    for name in ["abs", "sqrt", "exp", "log"] {
        cases.push((
            json!({"type":name,"dst":2,"src":0}),
            json!({"id":format!("core.{name}"),"params":{"source":"a","target":"answer"}}),
        ));
    }
    for name in ["eq", "ne", "lt", "le", "gt", "ge"] {
        cases.push((json!({"type":"compare","dst":2,"lhs":0,"rhs":1,"op":name}),json!({"id":"core.compare","params":{"left":"a","right":"b","op":name,"target":"answer"}})));
    }
    cases.push((
        json!({"type":"recip","dst":2,"src":0,"epsilon":0.001}),
        json!({"id":"core.recip","params":{"source":"a","target":"answer","epsilon":0.001}}),
    ));
    cases.push((
        json!({"type":"clip","dst":2,"src":0,"min":1.,"max":3.}),
        json!({"id":"core.clip","params":{"source":"a","target":"answer","min":1.,"max":3.}}),
    ));
    for method in ["sum", "max"] {
        cases.push((json!({"type":"normalize","dst":2,"src":0,"method":method,"epsilon":1e-9}),json!({"id":"core.normalize_values","params":{"source":"a","target":"answer","method":method,"epsilon":1e-9}})));
    }
    for operation in ["sum", "mean", "min", "max"] {
        cases.push((json!({"type":"reduce_scalar","dst":2,"src":0,"operation":operation}),json!({"id":"core.reduce_scalar","params":{"source":"a","target":"answer","op":operation}})));
    }
    cases
}
#[test]
fn numeric_instruction_matrix_matches_native_steps_on_both_backends_and_reopens() {
    for directed in [false, true] {
        for (instruction, native) in cases() {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("graph.journal");
            let initial = fixture(directed);
            let head = initial.current_state_id();
            let mut store = DurableGraph::create_from_graph(&path, &initial, None).unwrap();
            let source = store.read_view().unwrap();
            let p = pipeline(json!([
                {"id":"core.init_nodes","params":{"target":"a","value":4}},
                {"id":"core.attach_node_attr","params":{"source":"a","attr":"x","effective_at":10}},
                loop_step(plan(json!([
                    {"type":"load_node_attr","dst":0,"attr_name":"x","default":0.},
                    {"type":"load_scalar","dst":1,"value":2.},instruction,
                    {"type":"store_node_prop","src":2,"var_name":"answer"}
                ]),3,json!([])),2),
                {"id":"core.attach_node_attr","params":{"source":"answer","attr":"result","effective_at":20}}
            ]));
            let accepted = p
                .run_durable(&mut Context::new(), &mut store, &source)
                .unwrap();
            assert_eq!(accepted.write_state_id(), Some(head + 1));
            let memory = initial.into_shared();
            let selected = Subgraph::from_frozen_view(
                memory.clone(),
                memory.borrow().read_view().unwrap(),
                "native".into(),
            );
            // Native scalar reduction needs an explicit broadcast before attachment.
            let mut steps = vec![
                json!({"id":"core.init_nodes","params":{"target":"a","value":4}}),
                json!({"id":"core.init_scalar","params":{"target":"b","value":2.}}),
                native.clone(),
            ];
            if native["id"] == "core.reduce_scalar" {
                steps.push(json!({"id":"core.broadcast_scalar","params":{"scalar":"answer","reference":"a","target":"answer"}}));
            }
            steps.push(json!({"id":"core.attach_node_attr","params":{"source":"answer","attr":"result","effective_at":20}}));
            pipeline(json!(steps))
                .run(&mut Context::new(), selected)
                .unwrap();
            for node in [0, 2] {
                assert_eq!(
                    accepted.node_attr(node, &"result".into()).unwrap(),
                    memory
                        .borrow()
                        .get_node_attr(node, &"result".into())
                        .unwrap(),
                    "{instruction}"
                );
            }
            drop(store);
            let reopened = DurableGraph::open(&path).unwrap();
            assert_eq!(
                reopened
                    .read_view()
                    .unwrap()
                    .node_attr(2, &"result".into())
                    .unwrap(),
                accepted.node_attr(2, &"result".into()).unwrap()
            );
            assert_eq!(
                reopened
                    .read_view_at(15, None)
                    .unwrap()
                    .node_attr(0, &"result".into())
                    .unwrap(),
                None
            );
        }
    }
}
#[test]
fn exact_inputs_aliasing_register_isolation_and_simultaneous_carries() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
    let source = store.read_view().unwrap();
    let p = pipeline(json!([
        {"id":"core.init_nodes","params":{"target":"slot/0","value":9007199254740993i64}},
        loop_step(plan(json!([
            {"type":"load_node_prop","dst":0,"var_name":"slot/0"},
            {"type":"abs","dst":0,"src":0},
            {"type":"store_node_prop","src":0,"var_name":"exact"}
        ]),1,json!([])),1),
        loop_step(plan(json!([
            {"type":"store_node_prop","src":0,"var_name":"a"},
            {"type":"store_node_prop","src":1,"var_name":"b"},
            {"type":"init_nodes","dst":0,"value":1.},
            {"type":"init_nodes","dst":1,"value":2.}
        ]),2,json!([[0,1],[1,0]])),2),
        {"id":"core.attach_node_attr","params":{"source":"a","attr":"a"}},
        {"id":"core.attach_node_attr","params":{"source":"b","attr":"b"}},
        {"id":"core.attach_node_attr","params":{"source":"exact","attr":"exact"}},
        {"id":"core.attach_node_attr","params":{"source":"slot/0","attr":"unchanged"}}
    ]));
    let result = p
        .run_durable(&mut Context::new(), &mut store, &source)
        .unwrap();
    for node in [0, 2] {
        assert_eq!(
            result.node_attr(node, &"a".into()).unwrap(),
            Some(V::Float64(2.))
        );
        assert_eq!(
            result.node_attr(node, &"b".into()).unwrap(),
            Some(V::Float64(1.))
        );
        assert_eq!(
            result.node_attr(node, &"exact".into()).unwrap(),
            Some(V::Int(9007199254740993))
        );
        assert_eq!(
            result.node_attr(node, &"unchanged".into()).unwrap(),
            Some(V::Int(9007199254740993))
        );
    }
}
#[test]
fn failed_or_cancelled_lowered_program_never_publishes_prefix_writes() {
    for divisor in [0., 1e-20] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("graph.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
        let source = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
            {"id":"core.init_nodes","params":{"target":"prefix","value":9}},
            {"id":"core.attach_node_attr","params":{"source":"prefix","attr":"prefix"}},
            loop_step(plan(json!([
                {"type":"init_nodes","dst":0,"value":divisor},
                {"type":"load_scalar","dst":1,"value":divisor},
                {"type":"div","dst":0,"lhs":0,"rhs":1},
                {"type":"store_node_prop","src":0,"var_name":"answer"}
            ]),2,json!([])),1),
            {"id":"core.attach_node_attr","params":{"source":"answer","attr":"answer"}}
        ]));
        if divisor == 0. {
            let error = p
                .run_durable(&mut Context::new(), &mut store, &source)
                .unwrap_err();
            assert!(error.to_string().contains("division by zero"), "{error}");
            assert_eq!(fs::read(&path).unwrap(), bytes);
        } else {
            let cancel = Arc::new(AtomicBool::new(true));
            assert!(p
                .run_durable(&mut Context::with_cancel_token(cancel), &mut store, &source)
                .is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
            let result = p
                .run_durable(&mut Context::new(), &mut store, &source)
                .unwrap();
            assert_eq!(
                result.node_attr(0, &"answer".into()).unwrap(),
                Some(V::Float64(1.))
            );
        }
    }
}

#[test]
fn one_node_broadcast_where_counts_and_detached_worker_publish_once() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("one.journal");
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    let source = store.read_view().unwrap();
    let head = graph.current_state_id();
    let p = pipeline(json!([
        loop_step(plan(json!([
            {"type":"graph_node_count","dst":0},
            {"type":"broadcast_scalar","dst":0,"scalar":0},
            {"type":"graph_edge_count","dst":1},
            {"type":"broadcast_scalar","dst":1,"scalar":1},
            {"type":"where","dst":2,"condition":0,"if_true":0,"if_false":1},
            {"type":"store_node_prop","src":2,"var_name":"answer"}
        ]),3,json!([])),1),
        {"id":"core.attach_node_attr","params":{"source":"answer","attr":"answer"}}
    ]));
    let plan = p.prepare_durable(&store, &source).unwrap();
    let prepared = std::thread::scope(|scope| {
        scope
            .spawn(move || plan.execute(&mut Context::new()))
            .join()
            .unwrap()
    })
    .unwrap();
    let result = prepared.accept(&mut store).unwrap();
    assert_eq!(
        result.node_attr(0, &"answer".into()).unwrap(),
        Some(V::Int(1))
    );
    assert_eq!(result.write_state_id(), Some(head + 1));
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(
        reopened
            .read_view()
            .unwrap()
            .node_attr(0, &"answer".into())
            .unwrap(),
        Some(V::Int(1))
    );
}

#[test]
fn public_batch_executor_delegates_numeric_math_and_keeps_pointer_facade() {
    use groggy::algorithms::{
        execution::{BatchExecutor, BatchPlan},
        steps::{StepScope, StepVariables},
    };
    let owner = fixture(false).into_shared();
    let selected = Subgraph::from_frozen_view(
        owner.clone(),
        owner.borrow().read_view().unwrap(),
        "direct".into(),
    );
    let value = plan(
        json!([
            {"type":"init_nodes","dst":0,"value":1e-20},
            {"type":"load_scalar","dst":1,"value":1e-20},
            {"type":"div","dst":0,"lhs":0,"rhs":1},
            {"type":"store_node_prop","src":0,"var_name":"answer"}
        ]),
        2,
        json!([]),
    );
    let plan: BatchPlan = serde_json::from_value(value).unwrap();
    let mut executor = BatchExecutor::new(2);
    let mut variables = StepVariables::default();
    executor
        .execute(&plan, 1, &mut StepScope::new(&selected, &mut variables))
        .unwrap();
    assert_eq!(
        variables.node_map("answer").unwrap()[&2],
        groggy::algorithms::AlgorithmParamValue::Float(1.)
    );
    assert_eq!(executor.get_slot_pointers().len(), 2);
}

#[test]
fn pending_creation_selection_and_numeric_defaults_share_one_acceptance() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pending.journal");
    let mut store = DurableGraph::create(&path, Default::default(), None).unwrap();
    let source = store.read_view().unwrap();
    let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
        loop_step(plan(json!([
            {"type":"load_node_attr","dst":0,"attr_name":"missing","default":4.},
            {"type":"load_scalar","dst":1,"value":2.},
            {"type":"mul","dst":0,"lhs":0,"rhs":1},
            {"type":"store_node_prop","src":0,"var_name":"answer"}
        ]),2,json!([])),2),
        {"id":"core.attach_node_attr","params":{"source":"answer","attr":"answer","effective_at":20}}
    ]));
    let accepted = p
        .run_durable(&mut Context::new(), &mut store, &source)
        .unwrap();
    assert_eq!(accepted.node_ids(), vec![0]);
    assert_eq!(accepted.write_state_id(), Some(1));
    assert_eq!(
        accepted.node_attr(0, &"answer".into()).unwrap(),
        Some(V::Float64(8.))
    );
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert!(reopened
        .read_view_at(9, None)
        .unwrap()
        .node_ids()
        .is_empty());
    assert_eq!(
        reopened
            .read_view_at(15, None)
            .unwrap()
            .node_attr(0, &"answer".into())
            .unwrap(),
        None
    );
    assert_eq!(
        reopened
            .read_view_at(20, None)
            .unwrap()
            .node_attr(0, &"answer".into())
            .unwrap(),
        Some(V::Float64(8.))
    );
}

#[test]
fn direct_facade_preserves_list_intermediates_until_optional_pointer_projection() {
    use groggy::algorithms::{
        execution::{BatchExecutor, BatchPlan},
        steps::{StepScope, StepVariables},
    };
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    graph
        .set_node_attr(0, "choices".into(), V::IntVec(vec![5, 5, 7]))
        .unwrap();
    let owner = graph.into_shared();
    let selected = Subgraph::from_frozen_view(
        owner.clone(),
        owner.borrow().read_view().unwrap(),
        "lists".into(),
    );
    let instructions = plan(
        json!([
            {"type":"load_node_attr","dst":0,"attr_name":"choices","default":0.},
            {"type":"mode_list","dst":1,"src":0,"tie_break":"lowest"},
            {"type":"store_node_prop","src":1,"var_name":"answer"}
        ]),
        2,
        json!([]),
    );
    let plan: BatchPlan = serde_json::from_value(instructions).unwrap();
    let mut executor = BatchExecutor::new(1);
    let mut variables = StepVariables::default();
    executor
        .execute(&plan, 1, &mut StepScope::new(&selected, &mut variables))
        .unwrap();
    assert_eq!(
        variables.node_map("answer").unwrap()[&0],
        groggy::algorithms::AlgorithmParamValue::Int(5)
    );
    assert_eq!(executor.get_slot_pointers().len(), 2);
    assert_eq!(
        variables.node_map("answer").unwrap()[&0],
        groggy::algorithms::AlgorithmParamValue::Int(5)
    );
}
