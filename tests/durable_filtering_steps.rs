//! Filtering/order primitives retain exact domains and selected pending reads.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs};
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn fixture() -> Graph {
    let mut g = Graph::new_directed();
    let base = 1i64 << 53;
    g.apply_batch(
        GraphBatch {
            add_nodes: 6,
            node_attrs: [(
                "score".into(),
                [
                    (0, V::Int(base + 2)),
                    (2, V::Int(base)),
                    (3, V::Int(base + 1)),
                    (4, V::Null),
                    (5, V::Int(999)),
                ]
                .into_iter()
                .map(|(n, v)| (N::Added(n), v))
                .collect(),
            )]
            .into(),
            add_edges: [(0, 2), (2, 5), (3, 0), (3, 3)]
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
fn operations(order: &str) -> Value {
    let base = 1i64 << 53;
    json!([
     {"id":"core.filter_nodes_by_attr","params":{"attr":"score","target":"mask","predicate":{"type":"gt","value":{"type":"Float","value":base as f64}}}},
     {"id":"core.sort_nodes_by_attr","params":{"attr":"score","order":order,"target":"rank"}},
     {"id":"core.load_node_attr","params":{"attr":"score","target":"values"}},
     {"id":"core.top_k","params":{"source":"values","k":2,"order":order,"target":"top"}},
     {"id":"core.filter_edges_by_attr","params":{"attr":"weight","target":"edges","predicate":{"type":"gt","value":{"type":"Float","value":base as f64}}}},
     {"id":"core.attach_node_attr","params":{"source":"mask","attr":"mask","effective_at":20}},
     {"id":"core.attach_node_attr","params":{"source":"rank","attr":"rank","effective_at":20}},
     {"id":"core.attach_node_attr","params":{"source":"top","attr":"top","effective_at":20}},
     {"id":"core.attach_edge_attr","params":{"source":"edges","attr":"mask","effective_at":20}}
    ])
}
#[test]
fn all_filter_order_steps_match_exact_pending_results_memory_workers_and_recovery() {
    for order in ["asc", "desc"] {
        let graph = fixture();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("filter.journal");
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store
            .read_view()
            .unwrap()
            .select_nodes(&[0, 2, 3, 4])
            .unwrap();
        let head = selected.write_state_id().unwrap();
        let p = pipeline(json!([
         {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"score":(1i64<<53)+1},"effective_at":10}},
         {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"output":"new_edge","effective_at":10}},
         {"id":"core.init_edges","params":{"target":"weights","value":(1i64<<53)+1}},
         {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight","effective_at":15}},
         {"id":"iter.loop","params":{"iterations":2,"body":operations(order)}}
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
        let ordered = if order == "asc" {
            vec![2, 3, 6, 0, 4]
        } else {
            vec![0, 3, 6, 2, 4]
        };
        let ranks: BTreeMap<_, _> = ordered
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i as i64))
            .collect();
        for node in [0, 2, 3, 4, 6] {
            assert_eq!(
                accepted.node_attr(node, &"rank".into()).unwrap(),
                Some(V::Int(ranks[&node]))
            );
            assert_eq!(
                accepted.node_attr(node, &"mask".into()).unwrap(),
                if [0, 3, 6].contains(&node) {
                    Some(V::Int(1))
                } else {
                    None
                }
            );
            assert_eq!(
                accepted.node_attr(node, &"top".into()).unwrap(),
                if ranks[&node] < 2 {
                    Some(V::Int(ranks[&node]))
                } else {
                    None
                }
            );
        }
        for edge in [0, 2, 3, 4] {
            assert_eq!(
                accepted.edge_attr(edge, &"mask".into()).unwrap(),
                Some(V::Int(1))
            );
        }
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .edge_attr(1, &"mask".into())
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .node_attr(5, &"rank".into())
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
            for attr in ["mask", "rank", "top"] {
                assert_eq!(
                    owner.borrow().get_node_attr(node, &attr.into()).unwrap(),
                    accepted.node_attr(node, &attr.into()).unwrap()
                );
            }
        }
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        assert_eq!(
            reopened
                .read_view_at(19, None)
                .unwrap()
                .node_attr(0, &"rank".into())
                .unwrap(),
            None
        );
        assert_eq!(
            reopened
                .read_view_at(20, None)
                .unwrap()
                .node_attr(0, &"rank".into())
                .unwrap(),
            Some(V::Int(ranks[&0]))
        );
    }
}
#[test]
fn mixed_sort_domains_reject_even_with_zero_k_without_consuming_pending_ids() {
    for step in [
        json!({"id":"core.sort_nodes_by_attr","params":{"attr":"score","target":"bad"}}),
        json!({"id":"core.top_k","params":{"source":"values","k":0,"target":"bad"}}),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut graph = fixture();
        graph
            .set_node_attr(0, "score".into(), V::Text("bad".into()))
            .unwrap();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
         {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"score":0}}},
         {"id":"core.load_node_attr","params":{"attr":"score","target":"values","default":0}},step
        ]));
        assert!(p
            .run_durable(&mut Context::new(), &mut store, &selected)
            .is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
    }
}
#[test]
fn read_only_filter_order_and_topk_leave_history_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store
        .read_view()
        .unwrap()
        .select_nodes(&[0, 2, 3, 4])
        .unwrap();
    let bytes = fs::read(&path).unwrap();
    let p = pipeline(json!([
     {"id":"core.sort_nodes_by_attr","params":{"attr":"score","target":"rank"}},
     {"id":"core.top_k","params":{"source":"rank","k":0,"target":"empty"}},
     {"id":"core.filter_nodes_by_attr","params":{"attr":"score","target":"mask","predicate":{"type":"range","min":0.,"max":10.}}}
    ]));
    let result = p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .unwrap();
    assert_eq!(result.write_state_id(), selected.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), bytes);
}
