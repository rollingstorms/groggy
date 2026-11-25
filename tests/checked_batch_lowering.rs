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

/// Independent incidence oracle: no graph library or native step supplies the
/// expected direction, multiplicity, loop or asynchronous update result.
fn neighbor_rows(direction: &str) -> [Vec<usize>; 2] {
    match direction {
        "in" => [vec![1], vec![0, 0, 1]],
        "out" => [vec![1, 1], vec![0, 1]],
        "undirected" => [vec![1, 1, 1], vec![0, 0, 0, 1]],
        _ => panic!("test direction"),
    }
}
fn expected_aggregate(values: &[f64], op: &str) -> f64 {
    match op {
        "sum" => values.iter().sum(),
        "mean" => values.iter().sum::<f64>() / values.len() as f64,
        "min" => values.iter().copied().reduce(f64::min).unwrap(),
        "max" => values.iter().copied().reduce(f64::max).unwrap(),
        _ => panic!("test operation"),
    }
}
fn expected_mode(values: &[f64], tie: &str) -> f64 {
    let most = values
        .iter()
        .map(|a| values.iter().filter(|b| *b == a).count())
        .max()
        .unwrap();
    let candidates: Vec<_> = values
        .iter()
        .copied()
        .filter(|a| values.iter().filter(|b| *b == a).count() == most)
        .collect();
    match tie {
        "lowest" => candidates.into_iter().reduce(f64::min).unwrap(),
        "highest" => candidates.into_iter().reduce(f64::max).unwrap(),
        "first" => candidates[0],
        _ => panic!("test tie"),
    }
}
fn neighbor_fixture(directed: bool) -> Graph {
    let mut graph = fixture(directed);
    graph
        .apply_batch(
            GraphBatch {
                add_edges: vec![(N::Existing(2), N::Existing(0))],
                node_attrs: [(
                    "x".into(),
                    vec![(N::Existing(0), V::Int(2)), (N::Existing(2), V::Int(5))],
                )]
                .into(),
                effective_at: Some(0),
                ..Default::default()
            },
            graph.current_state_id(),
            "values".into(),
            "test".into(),
        )
        .unwrap();
    graph
}
fn as_float(value: V) -> f64 {
    match value {
        V::Int(i) => i as f64,
        V::Float64(f) => f,
        _ => panic!("unexpected result {value:?}"),
    }
}

#[test]
fn topology_and_fused_plans_match_incidence_oracles_and_recover_on_both_backends() {
    for directed in [false, true] {
        let mut cases: Vec<(Vec<Value>, Vec<Value>, [f64; 2])> = Vec::new();
        cases.push((
            vec![json!({"type":"node_degree","dst":0})],
            vec![json!({"id":"core.node_degree","params":{"target":"answer"}})],
            if directed { [2., 2.] } else { [3., 4.] },
        ));
        for direction in ["in", "out", "undirected"] {
            let rows = neighbor_rows(direction);
            for op in ["sum", "mean", "min", "max"] {
                let expected = std::array::from_fn(|i| {
                    expected_aggregate(
                        &rows[i].iter().map(|&n| [2., 5.][n]).collect::<Vec<_>>(),
                        op,
                    )
                });
                cases.push((vec![json!({"type":"neighbor_aggregate","src":0,"dst":0,"operation":op,"direction":direction})],
                    vec![json!({"id":"core.neighbor_agg","params":{"source":"a","target":"answer","agg":op,"direction":direction}})], expected));
                let native = if op == "sum" {
                    vec![
                        json!({"id":"graph.fused_neighbor_mul_agg","params":{"values":"a","scalars":"b","target":"answer","direction":direction}}),
                    ]
                } else {
                    vec![
                        json!({"id":"core.mul","params":{"left":"a","right":"b","target":"product"}}),
                        json!({"id":"core.neighbor_agg","params":{"source":"product","target":"answer","agg":op,"direction":direction}}),
                    ]
                };
                cases.push((vec![json!({"type":"fused_neighbor_mul_agg","src":0,"multiplier":1,"dst":0,"operation":op,"direction":direction})], native, expected.map(|v| v*2.)));
            }
            for tie in ["lowest", "highest", "first"] {
                let expected = std::array::from_fn(|i| {
                    expected_mode(
                        &rows[i].iter().map(|&n| [2., 5.][n]).collect::<Vec<_>>(),
                        tie,
                    )
                });
                // There is no separate public native direction-aware mode step;
                // direct lowering composes the collection and list-mode kernels.
                cases.push((vec![json!({"type":"neighbor_mode","src":0,"dst":0,"tie_break":tie,"direction":direction})], vec![], expected));
            }
        }
        for include_self in [false, true] {
            for tie in ["lowest", "highest", "first"] {
                let rows = neighbor_rows(if directed { "out" } else { "undirected" });
                let expected = std::array::from_fn(|i| {
                    let mut values = if include_self {
                        vec![[2., 5.][i]]
                    } else {
                        vec![]
                    };
                    values.extend(rows[i].iter().map(|&n| [2., 5.][n]));
                    expected_mode(&values, tie)
                });
                let native_tie = if tie == "first" { "keep" } else { tie };
                cases.push((vec![json!({"type":"collect_neighbor_values","src":0,"dst":0,"include_self":include_self}),
                    json!({"type":"mode_list","src":0,"dst":0,"tie_break":tie})],
                    vec![json!({"id":"core.collect_neighbor_values","params":{"source":"a","target":"list","include_self":include_self}}),
                        json!({"id":"core.mode_list","params":{"source":"list","target":"answer","tie_break":native_tie}})], expected));
                for ordered in [false, true] {
                    let mut expected = [2., 5.];
                    for i in 0..2 {
                        let mut values = if include_self {
                            vec![expected[i]]
                        } else {
                            vec![]
                        };
                        values.extend(rows[i].iter().map(|&n| expected[n]));
                        expected[i] = expected_mode(&values, tie);
                    }
                    cases.push((vec![json!({"type":"neighbor_mode_update","target":0,"include_self":include_self,"tie_break":tie,"ordered":ordered})],
                        vec![json!({"id":"core.neighbor_mode_update","params":{"target":"a","output":"answer","include_self":include_self,"tie_break":native_tie,"ordered":ordered}})], expected));
                }
            }
        }
        for ordered in [false, true] {
            cases.push((vec![json!({"type":"update_in_place","source":0,"target":1,"ordered":ordered}),
                json!({"type":"update_in_place","source":1,"target":0,"ordered":ordered})],
                vec![json!({"id":"core.update_in_place","params":{"source":"a","target":"b","output":"answer","ordered":ordered}})], [2.,5.]));
        }
        cases.push((vec![json!({"type":"fused_m_a_d_d","dst":0,"a":0,"b":1,"c":2})],
            vec![json!({"id":"core.fused_madd","params":{"a":"a","b":"b","c":"c","target":"answer"}})], [7.,13.]));
        cases.push((vec![json!({"type":"load_scalar","dst":1,"value":2.}),json!({"type":"fused_a_x_p_y","dst":0,"alpha":1,"x":0,"y":2})],
            vec![json!({"id":"core.init_scalar","params":{"target":"alpha","value":2.}}),
                json!({"id":"core.init_scalar","params":{"target":"one","value":1.}}),
                json!({"id":"core.fused_axpy","params":{"a":"alpha","x":"a","b":"one","y":"c","target":"answer"}})], [7.,13.]));
        for (index, (body, native, expected)) in cases.into_iter().enumerate() {
            let initial = neighbor_fixture(directed);
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("neighbors.journal");
            let mut store = DurableGraph::create_from_graph(&path, &initial, None).unwrap();
            let source = store.read_view().unwrap();
            let mut instructions = vec![
                json!({"type":"load_node_attr","dst":0,"attr_name":"x","default":0.}),
                json!({"type":"init_nodes","dst":1,"value":2.}),
                json!({"type":"init_nodes","dst":2,"value":3.}),
            ];
            instructions.extend(body);
            instructions.push(json!({"type":"store_node_prop","src":0,"var_name":"answer"}));
            let p = pipeline(
                json!([loop_step(plan(json!(instructions), 3, json!([])), 2),
                {"id":"core.attach_node_attr","params":{"source":"answer","attr":"result","effective_at":20}}]),
            );
            let accepted = p
                .run_durable(&mut Context::new(), &mut store, &source)
                .unwrap_or_else(|e| panic!("case {index}, directed={directed}: {e:#}"));
            assert_eq!(
                accepted.write_state_id(),
                Some(initial.current_state_id() + 1)
            );
            let owner = initial.into_shared();
            let selection = Subgraph::from_frozen_view(
                owner.clone(),
                owner.borrow().read_view().unwrap(),
                "memory".into(),
            );
            p.run(&mut Context::new(), selection).unwrap();
            for (row, node) in [0, 2].into_iter().enumerate() {
                assert_eq!(
                    as_float(accepted.node_attr(node, &"result".into()).unwrap().unwrap()),
                    expected[row],
                    "case {index}, directed={directed}, row={row}"
                );
                assert_eq!(
                    accepted.node_attr(node, &"result".into()).unwrap(),
                    owner
                        .borrow()
                        .get_node_attr(node, &"result".into())
                        .unwrap()
                );
            }
            if !native.is_empty() {
                let native_owner = neighbor_fixture(directed).into_shared();
                let selected = Subgraph::from_frozen_view(
                    native_owner.clone(),
                    native_owner.borrow().read_view().unwrap(),
                    "native".into(),
                );
                let mut steps = vec![
                    json!({"id":"core.load_node_attr","params":{"target":"a","attr":"x","default":0.}}),
                    json!({"id":"core.init_nodes","params":{"target":"b","value":2.}}),
                    json!({"id":"core.init_nodes","params":{"target":"c","value":3.}}),
                ];
                steps.extend(native);
                steps.push(json!({"id":"core.attach_node_attr","params":{"source":"answer","attr":"result"}}));
                pipeline(json!(steps))
                    .run(&mut Context::new(), selected)
                    .unwrap();
                for node in [0, 2] {
                    assert_eq!(
                        accepted.node_attr(node, &"result".into()).unwrap(),
                        native_owner
                            .borrow()
                            .get_node_attr(node, &"result".into())
                            .unwrap()
                    );
                }
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
                    .read_view_at(19, None)
                    .unwrap()
                    .node_attr(2, &"result".into())
                    .unwrap(),
                None
            );
        }
    }
}

#[test]
fn pending_endpoint_revisions_and_removals_feed_the_same_checked_topology() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pending-topology.journal");
    let initial = neighbor_fixture(true);
    let mut store = DurableGraph::create_from_graph(&path, &initial, None).unwrap();
    let source = store.read_view().unwrap();
    // Old edge 0 is 0 -> 2. Revise it to a self-loop and remove its parallel
    // sibling. The pending graph has 0->0, 2->2 and 2->0.
    let p = pipeline(json!([
        {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,0]],"effective_at":10}},
        {"id":"graph.remove_edges","params":{"ids":[1],"effective_at":10}},
        loop_step(plan(json!([
            {"type":"node_degree","dst":0},
            {"type":"neighbor_aggregate","src":0,"dst":1,"operation":"sum","direction":"in"},
            {"type":"store_node_prop","src":0,"var_name":"degree"},
            {"type":"store_node_prop","src":1,"var_name":"sum"}
        ]),2,json!([])),1),
        {"id":"core.attach_node_attr","params":{"source":"degree","attr":"degree","effective_at":20}},
        {"id":"core.attach_node_attr","params":{"source":"sum","attr":"sum","effective_at":20}}
    ]));
    let accepted = p
        .run_durable(&mut Context::new(), &mut store, &source)
        .unwrap();
    assert_eq!(accepted.edge_endpoints(0).unwrap(), (0, 0));
    assert_eq!(source.edge_endpoints(0).unwrap(), (0, 2));
    assert_eq!(accepted.edge_ids(), vec![0, 2, 3]);
    for (node, degree, sum) in [(0, 1, 3.), (2, 2, 2.)] {
        assert_eq!(
            accepted.node_attr(node, &"degree".into()).unwrap(),
            Some(V::Int(degree))
        );
        assert_eq!(
            accepted.node_attr(node, &"sum".into()).unwrap(),
            Some(V::Float64(sum))
        );
    }
    let owner = initial.into_shared();
    let selected = Subgraph::from_frozen_view(
        owner.clone(),
        owner.borrow().read_view().unwrap(),
        "pending".into(),
    );
    p.run(&mut Context::new(), selected).unwrap();
    assert_eq!(
        owner
            .borrow()
            .read_view()
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 0)
    );
    assert_eq!(
        owner.borrow().get_node_attr(0, &"sum".into()).unwrap(),
        Some(V::Float64(3.))
    );
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(
        reopened
            .read_view_at(9, None)
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 2)
    );
    assert_eq!(
        reopened
            .read_view_at(10, None)
            .unwrap()
            .edge_endpoints(0)
            .unwrap(),
        (0, 0)
    );
    assert_eq!(
        reopened
            .read_view_at(19, None)
            .unwrap()
            .node_attr(0, &"sum".into())
            .unwrap(),
        None
    );
}

#[test]
fn large_integer_neighbor_modes_are_exact_and_fused_overflow_never_publishes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("labels.journal");
    let mut graph = neighbor_fixture(true);
    let low = 9_007_199_254_740_992i64;
    let high = low + 1;
    graph
        .apply_batch(
            GraphBatch {
                node_attrs: [(
                    "label".into(),
                    vec![
                        (N::Existing(0), V::Int(high)),
                        (N::Existing(2), V::Int(low)),
                    ],
                )]
                .into(),
                ..Default::default()
            },
            graph.current_state_id(),
            "labels".into(),
            "test".into(),
        )
        .unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    for tie in ["lowest", "highest", "first"] {
        let source = store.read_view().unwrap();
        let p = pipeline(json!([
            loop_step(plan(json!([
                {"type":"load_node_attr","dst":0,"attr_name":"label","default":0.},
                {"type":"neighbor_mode","src":0,"dst":0,"tie_break":tie,"direction":"out"},
                {"type":"store_node_prop","src":0,"var_name":"result"}
            ]),1,json!([])),1),
            {"id":"core.attach_node_attr","params":{"source":"result","attr":"mode"}}
        ]));
        let result = p
            .run_durable(&mut Context::new(), &mut store, &source)
            .unwrap();
        assert_eq!(
            result.node_attr(2, &"mode".into()).unwrap(),
            Some(V::Int(if tie == "lowest" { low } else { high }))
        );
    }
    for operation in [
        json!({"type":"fused_madd","dst":0,"a":0,"b":1,"c":2}),
        json!({"type":"fused_axpy","dst":0,"alpha":1,"x":0,"y":2}),
        json!({"type":"fused_neighbor_mul_agg","dst":0,"src":0,"multiplier":1,"operation":"sum","direction":"in"}),
    ] {
        let source = store.read_view().unwrap();
        let before = fs::read(&path).unwrap();
        let head = store.graph().unwrap().current_state_id();
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            loop_step(plan(json!([
                {"type":"init_nodes","dst":0,"value":1e308},
                {"type":"init_nodes","dst":1,"value":1e308},
                {"type":"init_nodes","dst":2,"value":0.},operation,
                {"type":"store_node_prop","src":0,"var_name":"result"}
            ]),3,json!([])),1),
            {"id":"core.attach_node_attr","params":{"source":"result","attr":"bad"}}
        ]));
        assert!(format!(
            "{:#}",
            p.run_durable(&mut Context::new(), &mut store, &source)
                .unwrap_err()
        )
        .contains("not finite"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(store.graph().unwrap().current_state_id(), head);
    }
}

#[test]
fn mode_list_rejects_incomparable_ties_and_nonfinite_lists_without_installing() {
    use groggy::algorithms::{
        steps::{ModeListStep, ModeTieBreak, Step, StepScope, StepVariables},
        AlgorithmParamValue as A,
    };
    let owner = neighbor_fixture(true).into_shared();
    let selected = Subgraph::from_frozen_view(
        owner.clone(),
        owner.borrow().read_view().unwrap(),
        "mode".into(),
    );
    // Equal numeric labels of different representations use a stable key.
    for tie in [ModeTieBreak::Lowest, ModeTieBreak::Highest] {
        for _ in 0..20 {
            let mut vars = StepVariables::default();
            vars.set_node_map("source", [(0, A::Json(json!([1.0, 1])))].into());
            ModeListStep::new("source", "target", tie)
                .apply(
                    &mut Context::new(),
                    &mut StepScope::new(&selected, &mut vars),
                )
                .unwrap();
            assert_eq!(vars.node_map("target").unwrap()[&0], A::Int(1));
        }
    }
    for value in [A::Json(json!([1, "one"])), A::FloatList(vec![1., f64::NAN])] {
        let mut vars = StepVariables::default();
        vars.set_node_map("source", [(0, value)].into());
        vars.set_node_map("target", [(0, A::Int(17))].into());
        let step = ModeListStep::new("source", "target", ModeTieBreak::Lowest);
        assert!(step
            .apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut vars)
            )
            .is_err());
        assert_eq!(vars.node_map("target").unwrap()[&0], A::Int(17));
    }
}

#[test]
fn isolated_nodes_keep_native_empty_neighborhood_results() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("isolated.journal");
    let mut graph = Graph::new();
    graph.add_node().unwrap();
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    let source = store.read_view().unwrap();
    let p = pipeline(json!([
        {"id":"core.init_nodes","params":{"target":"values","value":3}},
        {"id":"core.node_degree","params":{"target":"degree"}},
        {"id":"core.neighbor_agg","params":{"source":"values","target":"sum","agg":"sum"}},
        {"id":"core.neighbor_agg","params":{"source":"values","target":"min","agg":"min"}},
        {"id":"core.collect_neighbor_values","params":{"source":"values","target":"list","include_self":false}},
        {"id":"core.mode_list","params":{"source":"list","target":"mode"}},
        {"id":"core.attach_node_attr","params":{"source":"degree","attr":"degree"}},
        {"id":"core.attach_node_attr","params":{"source":"sum","attr":"sum"}},
        {"id":"core.attach_node_attr","params":{"source":"min","attr":"min"}},
        {"id":"core.attach_node_attr","params":{"source":"mode","attr":"mode"}}
    ]));
    let result = p
        .run_durable(&mut Context::new(), &mut store, &source)
        .unwrap();
    assert_eq!(
        result.node_attr(0, &"degree".into()).unwrap(),
        Some(V::Int(0))
    );
    assert_eq!(
        result.node_attr(0, &"sum".into()).unwrap(),
        Some(V::Float64(0.))
    );
    assert_eq!(result.node_attr(0, &"min".into()).unwrap(), Some(V::Null));
    assert_eq!(result.node_attr(0, &"mode".into()).unwrap(), Some(V::Null));
}

#[test]
fn integer_mode_specialization_matches_general_labels_at_signed_limits() {
    use groggy::algorithms::{
        steps::{ModeListStep, ModeTieBreak, Step, StepScope, StepVariables},
        AlgorithmParamValue as A,
    };
    let owner = neighbor_fixture(true).into_shared();
    let selected = Subgraph::from_frozen_view(
        owner.clone(),
        owner.borrow().read_view().unwrap(),
        "integer mode".into(),
    );
    for (low, high) in [
        (i64::MIN, i64::MIN + 1),
        (i64::MAX - 1, i64::MAX),
        (9_007_199_254_740_992, 9_007_199_254_740_993),
    ] {
        for values in [
            vec![high, low, low, high],
            vec![low, high, high, low],
            vec![high, high, high, low, low],
        ] {
            for tie in [
                ModeTieBreak::Lowest,
                ModeTieBreak::Highest,
                ModeTieBreak::Keep,
            ] {
                let mut mixed: Vec<Value> = values.iter().map(|value| json!(*value)).collect();
                // This nonmodal floating category forces the general kernel but
                // cannot change either modal frequency or first modal occurrence.
                mixed.push(json!(1.5));
                let mut vars = StepVariables::default();
                vars.set_node_map("typed", [(0, A::IntList(values.clone()))].into());
                vars.set_node_map("json", [(0, A::Json(json!(values)))].into());
                vars.set_node_map("general", [(0, A::Json(json!(mixed)))].into());
                for name in ["typed", "json", "general"] {
                    ModeListStep::new(name, format!("{name}/result"), tie)
                        .apply(
                            &mut Context::new(),
                            &mut StepScope::new(&selected, &mut vars),
                        )
                        .unwrap();
                }
                let expected = if values.iter().filter(|&&value| value == high).count() == 3 {
                    high
                } else {
                    match tie {
                        ModeTieBreak::Lowest => low,
                        ModeTieBreak::Highest => high,
                        ModeTieBreak::Keep => values[0],
                    }
                };
                for name in ["typed", "json", "general"] {
                    assert_eq!(
                        vars.node_map(&format!("{name}/result")).unwrap()[&0],
                        A::Int(expected)
                    );
                }
            }
        }
    }
}
