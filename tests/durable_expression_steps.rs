//! One evaluator serves legacy expressions and detached selected read inputs.
use groggy::{
    algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
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
            node_attrs: [(
                "score".into(),
                [1, 999, 2, 3, 4, 999]
                    .into_iter()
                    .enumerate()
                    .map(|(n, v)| (N::Added(n), V::Int(v)))
                    .collect(),
            )]
            .into(),
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
    g
}
fn expr(kind: usize) -> Value {
    let current = json!({"type":"var","name":"value"});
    let score = json!({"type":"attr","name":"score"});
    match kind {
        0 => json!({"type":"binary_op","op":"add","left":current,"right":score}),
        1 => {
            json!({"type":"binary_op","op":"add","left":score,"right":{"type":"call","func":"sum","args":[{"type":"call","func":"neighbor_values","args":[{"type":"var","name":"values"}]}]}})
        }
        2 => {
            json!({"type":"binary_op","op":"add","left":current,"right":{"type":"call","func":"neighbor_count","args":[]}})
        }
        _ => {
            json!({"type":"if","condition":{"type":"const","value":false},"then_expr":{"type":"binary_op","op":"div","left":{"type":"const","value":1},"right":{"type":"const","value":0}},"else_expr":{"type":"binary_op","op":"add","left":current,"right":{"type":"const","value":2}}})
        }
    }
}
fn expected(directed: bool, asynchronous: bool, kind: usize) -> BTreeMap<usize, i64> {
    let mut values: BTreeMap<_, _> = [(0, 1), (3, 3), (4, 4), (6, 6)].into();
    let edges = [(3, 0), (3, 3), (3, 6), (0, 6), (6, 4), (0, 6)];
    for _ in 0..2 {
        let attrs = values.clone();
        let mut output = BTreeMap::new();
        for node in [0, 3, 4, 6] {
            let mut neighbors = Vec::new();
            for (a, b) in edges {
                if a == node {
                    neighbors.push(b);
                }
                if !directed && b == node && a != b {
                    neighbors.push(a);
                }
            }
            let source = if asynchronous { &values } else { &attrs };
            let result = match kind {
                0 => source[&node] + attrs[&node],
                1 => {
                    attrs[&node]
                        + neighbors
                            .into_iter()
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .map(|n| source[&n])
                            .sum::<i64>()
                }
                2 => source[&node] + neighbors.len() as i64,
                _ => source[&node] + 2,
            };
            if asynchronous {
                values.insert(node, result);
            } else {
                output.insert(node, result);
            }
        }
        if !asynchronous {
            values = output;
        }
    }
    values
}
#[test]
fn expression_matrix_preserves_pending_attributes_neighbors_async_order_and_recovery() {
    for directed in [false, true] {
        for asynchronous in [false, true] {
            for kind in 0..4 {
                let graph = fixture(directed);
                let temp = tempfile::tempdir().unwrap();
                let path = temp.path().join("expr.journal");
                let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
                let source = store
                    .read_view()
                    .unwrap()
                    .select_nodes(&[0, 2, 3, 4])
                    .unwrap();
                let head = source.write_state_id().unwrap();
                let p = pipeline(json!([
                    {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"score":6},"effective_at":10}},
                    {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}],[{"added":0},4],[0,{"added":0}]],"output":"edges","effective_at":10}},
                    {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,3,{"added":0}]],"effective_at":10}},
                    {"id":"graph.remove_nodes","params":{"ids":[2],"effective_at":10}},
                    {"id":"core.load_node_attr","params":{"attr":"score","target":"values"}},
                    {"id":"iter.loop","params":{"iterations":2,"body":[
                        {"id":"core.map_nodes","params":{"source":"values","target":"values","async_update":asynchronous,"expr":expr(kind)}},
                        {"id":"core.attach_node_attr","params":{"source":"values","attr":"score","effective_at":20}}
                    ]}}
                ]));
                let plan = p.prepare_durable(&store, &source).unwrap();
                let pending = std::thread::scope(|threads| {
                    threads
                        .spawn(move || plan.execute(&mut Context::new()).unwrap())
                        .join()
                        .unwrap()
                });
                let accepted = pending.accept(&mut store).unwrap();
                assert_eq!(accepted.write_state_id(), Some(head + 1));
                for (node, value) in expected(directed, asynchronous, kind) {
                    assert_eq!(
                        accepted.node_attr(node, &"score".into()).unwrap(),
                        Some(V::Int(value)),
                        "{directed}/{asynchronous}/{kind}/{node}"
                    );
                }
                assert_eq!(
                    store
                        .read_view()
                        .unwrap()
                        .node_attr(5, &"score".into())
                        .unwrap(),
                    Some(V::Int(999))
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
                        owner.borrow().get_node_attr(node, &"score".into()).unwrap(),
                        accepted.node_attr(node, &"score".into()).unwrap()
                    );
                }
                drop(store);
                let reopened = DurableGraph::open(&path).unwrap();
                assert_eq!(
                    reopened
                        .read_view_at(19, None)
                        .unwrap()
                        .node_attr(0, &"score".into())
                        .unwrap(),
                    Some(V::Int(1))
                );
                assert_eq!(
                    reopened
                        .read_view_at(20, None)
                        .unwrap()
                        .node_attr(0, &"score".into())
                        .unwrap(),
                    accepted.node_attr(0, &"score".into()).unwrap()
                );
            }
        }
    }
}
#[test]
fn late_expression_failure_discards_local_async_values_outputs_and_pending_allocations() {
    for asynchronous in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(true), None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"score":5}}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.emit","params":{"source":"pending","name":"discard"}},
            {"id":"core.load_node_attr","params":{"attr":"score","target":"values"}},
            {"id":"core.map_nodes","params":{"source":"values","target":"values","async_update":asynchronous,"expr":{"type":"binary_op","op":"div","left":{"type":"const","value":1},"right":{"type":"binary_op","op":"sub","left":{"type":"var","name":"value"},"right":{"type":"const","value":3}}}}}
        ]));
        let mut ctx = Context::new();
        ctx.add_output("retained", AlgorithmOutput::GraphView(selected.clone()));
        assert!(p.run_durable(&mut ctx, &mut store, &selected).is_err());
        assert_eq!(ctx.take_outputs().len(), 1);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
    }
}
