//! Existing structural kernels share pending immutable reads across backends.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::fs;
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
            add_edges: [(0, 2), (2, 3), (3, 0), (0, 2), (3, 3), (3, 5)]
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
#[test]
fn all_structural_steps_see_staged_topology_weights_and_match_memory_and_recovery() {
    let graph = fixture(false);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("structural.journal");
    let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
    let selected = store
        .read_view()
        .unwrap()
        .select_nodes(&[0, 2, 3, 4])
        .unwrap();
    let head = graph.current_state_id();
    let mut body = Vec::new();
    for (name, options) in [
        ("weighted_degree", json!({"weight_attr":"weight"})),
        ("edge_weight_sum", json!({"weight_attr":"weight"})),
        ("triangle_count", json!({})),
        ("k_core_mark", json!({"k":2})),
    ] {
        let mut params = options;
        params["target"] = json!(name);
        body.push(json!({"id":format!("core.{name}"),"params":params}));
        body.push(json!({"id":"core.attach_node_attr","params":{"source":name,"attr":name,"effective_at":30}}));
    }
    let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
        {"id":"graph.add_edges","params":{"edges":[[3,{"added":0}]],"output":"edge","effective_at":15}},
        {"id":"graph.set_edge_endpoints","params":{"revisions":[[3,4,{"added":0}]],"effective_at":15}},
        {"id":"core.init_edges","params":{"value":2.5,"target":"weights"}},
        {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight","effective_at":20}},
        {"id":"iter.loop","params":{"iterations":2,"body":body}}
    ]));
    let plan = p.prepare_durable(&store, &selected).unwrap();
    let prepared = std::thread::scope(|threads| {
        threads
            .spawn(move || plan.execute(&mut Context::new()).unwrap())
            .join()
            .unwrap()
    });
    let accepted = prepared.accept(&mut store).unwrap();
    assert_eq!(accepted.node_ids(), vec![0, 2, 3, 4, 6]);
    assert_eq!(accepted.write_state_id(), Some(head + 1));
    for (node, weight, triangle, core) in [
        (0, 5., 1, 1),
        (2, 5., 1, 1),
        (3, 12.5, 1, 1),
        (4, 2.5, 0, 0),
        (6, 5., 0, 0),
    ] {
        for attr in ["weighted_degree", "edge_weight_sum"] {
            assert_eq!(
                accepted.node_attr(node, &attr.into()).unwrap(),
                Some(V::Float64(weight)),
                "{attr}/{node}"
            );
        }
        assert_eq!(
            accepted.node_attr(node, &"triangle_count".into()).unwrap(),
            Some(V::Int(triangle))
        );
        assert_eq!(
            accepted.node_attr(node, &"k_core_mark".into()).unwrap(),
            Some(V::Int(core))
        );
    }
    assert_eq!(
        store
            .read_view()
            .unwrap()
            .node_attr(5, &"weighted_degree".into())
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
        for attr in [
            "weighted_degree",
            "edge_weight_sum",
            "triangle_count",
            "k_core_mark",
        ] {
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
            .read_view_at(29, None)
            .unwrap()
            .node_attr(3, &"weighted_degree".into())
            .unwrap(),
        None
    );
    assert_eq!(
        reopened
            .read_view_at(30, None)
            .unwrap()
            .node_attr(3, &"weighted_degree".into())
            .unwrap(),
        Some(V::Float64(12.5))
    );
}
#[test]
fn directed_structural_domains_and_bad_weights_discard_the_entire_program() {
    for (directed, name, options, weight) in [
        (true, "triangle_count", json!({}), json!(2.)),
        (true, "k_core_mark", json!({"k":2}), json!(2.)),
        (
            false,
            "weighted_degree",
            json!({"weight_attr":"weight"}),
            json!("bad"),
        ),
        (
            false,
            "edge_weight_sum",
            json!({"weight_attr":"weight"}),
            json!(1e308),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(directed), None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let head = selected.write_state_id();
        let mut options = options;
        options["target"] = json!("answer");
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"core.init_edges","params":{"value":weight,"target":"weights"}},
            {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}},
            {"id":format!("core.{name}"),"params":options}
        ]));
        assert!(
            p.run_durable(&mut Context::new(), &mut store, &selected)
                .is_err(),
            "{name}"
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(store.read_view().unwrap().write_state_id(), head);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
    }
}
