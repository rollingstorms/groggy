//! Prepared message-passing blocks preserve their established update semantics.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs};
fn pipeline(steps: Value) -> Pipeline {
    try_pipeline(steps).unwrap()
}
fn try_pipeline(steps: Value) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
}
fn block(direction: &str, ordered: bool, include_self: bool, body: Value) -> Value {
    json!({"id":"core.execution_block","params":{"mode":"message_pass","target":"values","options":{"ordered":ordered,"include_self":include_self,"direction":direction},"body":{"nodes":body}}})
}
fn fixture(directed: bool) -> Graph {
    let mut g = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    g.apply_batch(
        GraphBatch {
            add_nodes: 6,
            add_edges: [(0, 2), (2, 3), (3, 0), (0, 2), (3, 3), (5, 0)]
                .into_iter()
                .map(|(a, b)| (N::Added(a), N::Added(b)))
                .collect(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "test".into(),
    )
    .unwrap();
    g.apply_batch(
        GraphBatch {
            remove_nodes: vec![N::Existing(1)],
            effective_at: Some(0),
            ..Default::default()
        },
        g.current_state_id(),
        "sparse".into(),
        "test".into(),
    )
    .unwrap();
    g
}
fn expected(direction: &str, ordered: bool, include_self: bool) -> BTreeMap<usize, f64> {
    let nodes = [0, 2, 3, 4, 6];
    let edges = [(0, 6), (2, 3), (3, 0), (0, 2), (3, 3), (4, 6), (0, 2)];
    let mut values: BTreeMap<_, _> = nodes
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i as f64))
        .collect();
    for _ in 0..2 {
        let before = values.clone();
        let mut after = BTreeMap::new();
        for node in nodes {
            let mut neighbors = Vec::new();
            for (s, t) in edges {
                if (direction == "out" || direction == "undirected") && s == node {
                    neighbors.push(t);
                }
                if (direction == "in" || direction == "undirected")
                    && t == node
                    && !(direction == "undirected" && s == t)
                {
                    neighbors.push(s);
                }
            }
            if include_self && !neighbors.contains(&node) {
                neighbors.push(node);
            }
            neighbors.sort_unstable();
            let source = if ordered { &values } else { &before };
            let sum = neighbors.into_iter().map(|n| source[&n]).sum();
            if ordered {
                values.insert(node, sum);
            } else {
                after.insert(node, sum);
            }
        }
        if !ordered {
            values = after;
        }
    }
    values
}
#[test]
fn message_pass_matrix_matches_independent_updates_memory_workers_and_reopen() {
    for directed in [false, true] {
        for direction in ["in", "out", "undirected"] {
            for ordered in [false, true] {
                for include_self in [false, true] {
                    let graph = fixture(directed);
                    let temp = tempfile::tempdir().unwrap();
                    let path = temp.path().join("block.journal");
                    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
                    let selected = store
                        .read_view()
                        .unwrap()
                        .select_nodes(&[0, 2, 3, 4])
                        .unwrap();
                    let head = selected.write_state_id().unwrap();
                    let body = json!([{"id":"sum","domain":"core","op_type":"neighbor_agg","inputs":["values"],"output":"sum","metadata":{"agg":"sum"}}]);
                    let p = pipeline(json!([
                        {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
                        {"id":"graph.add_edges","params":{"edges":[[4,{"added":0}],[0,2]],"output":"edges","effective_at":15}},
                        {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,{"added":0}]],"effective_at":15}},
                        {"id":"core.init_nodes_with_index","params":{"target":"values"}},
                        {"id":"iter.loop","params":{"iterations":2,"body":[block(direction,ordered,include_self,body)]}},
                        {"id":"core.attach_node_attr","params":{"source":"values","attr":"result","effective_at":30}}
                    ]));
                    let plan = p.prepare_durable(&store, &selected).unwrap();
                    let pending = std::thread::scope(|threads| {
                        threads
                            .spawn(move || plan.execute(&mut Context::new()).unwrap())
                            .join()
                            .unwrap()
                    });
                    let accepted = pending.accept(&mut store).unwrap();
                    assert_eq!(accepted.write_state_id(), Some(head + 1));
                    for (node, value) in expected(direction, ordered, include_self) {
                        assert_eq!(
                            accepted.node_attr(node, &"result".into()).unwrap(),
                            Some(V::Float64(value)),
                            "{directed}/{direction}/{ordered}/{include_self}/{node}"
                        );
                    }
                    assert_eq!(
                        store
                            .read_view()
                            .unwrap()
                            .node_attr(5, &"result".into())
                            .unwrap(),
                        None
                    );
                    let owner = graph.into_shared();
                    let memory = Subgraph::from_frozen_view(
                        owner.clone(),
                        owner
                            .borrow()
                            .read_view()
                            .unwrap()
                            .select_nodes(&[0, 2, 3, 4])
                            .unwrap(),
                        "test".into(),
                    );
                    p.run(&mut Context::new(), memory).unwrap();
                    for node in accepted.node_ids() {
                        assert_eq!(
                            owner
                                .borrow()
                                .get_node_attr(node, &"result".into())
                                .unwrap(),
                            accepted.node_attr(node, &"result".into()).unwrap()
                        );
                    }
                    drop(store);
                    let reopened = DurableGraph::open(&path).unwrap();
                    assert_eq!(
                        reopened
                            .read_view_at(29, None)
                            .unwrap()
                            .node_attr(0, &"result".into())
                            .unwrap(),
                        None
                    );
                    assert_eq!(
                        reopened
                            .read_view_at(30, None)
                            .unwrap()
                            .node_attr(0, &"result".into())
                            .unwrap(),
                        accepted.node_attr(0, &"result".into()).unwrap()
                    );
                }
            }
        }
    }
}
#[test]
fn unsupported_block_modes_and_bodies_reject_before_a_prefix_even_in_zero_loops() {
    for (mode, body) in [
        ("streaming", json!([])),
        (
            "message_pass",
            json!([{"id":"bad","domain":"core","op_type":"not_implemented","inputs":[],"output":"bad"}]),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let store = DurableGraph::create_from_graph(&path, &fixture(true), None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let mut unsupported = block("in", true, false, body);
        unsupported["params"]["mode"] = json!(mode);
        let p = try_pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"iter.loop","params":{"iterations":0,"body":[unsupported]}}
        ]));
        if let Ok(p) = p {
            assert!(p.prepare_durable(&store, &selected).is_err());
        }
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}
#[test]
fn a_late_checked_arithmetic_error_discards_prior_staged_ids_and_writes() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("overflow.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(true), None).unwrap();
    let selected = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let body = json!([
        {"id":"large","domain":"core","op_type":"constant","inputs":[],"output":"large","metadata":{"value":i64::MAX}},
        {"id":"add","domain":"core","op_type":"add","inputs":["values","large"],"output":"answer"}
    ]);
    let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"core.init_nodes_with_index","params":{"target":"values"}},
        {"id":"core.attach_node_attr","params":{"source":"values","attr":"prefix"}},
        block("in",false,false,body)
    ]));
    assert!(p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .unwrap_err()
        .to_string()
        .contains("overflow"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
}
