//! Selected shortest/Yen/walk primitives share pending read authority on workers.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{collections::HashSet, fs};
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
            add_edges: [(0, 2), (0, 2), (2, 2), (2, 4), (0, 4), (4, 5)]
                .into_iter()
                .map(|(a, b)| (N::Added(a), N::Added(b)))
                .collect(),
            edge_attrs: [(
                "w".into(),
                [5., 2., 0., 1., 10., 1.]
                    .into_iter()
                    .enumerate()
                    .map(|(i, w)| (E::Added(i), V::Float64(w)))
                    .collect(),
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
    g
}
#[test]
fn path_primitives_match_independent_paths_pending_workers_memory_and_recovery() {
    for directed in [false, true] {
        for weighted in [false, true] {
            let graph = fixture(directed);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("paths.journal");
            let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
            let source = store.read_view().unwrap().select_nodes(&[0, 2, 4]).unwrap();
            let weight = if weighted { json!("w") } else { Value::Null };
            let mut shortest = json!({"id":"core.shortest_path_map","params":{"source":"start","output":"distances"}});
            let mut paths = json!({"id":"core.k_shortest_paths","params":{"source":"start","target":"end","k":10,"output":"paths"}});
            let mut walk = json!({"id":"core.random_walk","params":{"start_nodes":"start","length":6,"restart_prob":0.2,"seed":19,"output":"walk"}});
            if weighted {
                for step in [&mut shortest, &mut paths, &mut walk] {
                    step["params"]["weight_attr"] = weight.clone();
                }
            }
            let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[4,{"added":0}]],"attributes":{"w":2.0},"output":"edge","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"attributes":{"w":8.0},"output":"direct","effective_at":10}},
            {"id":"graph.remove_edges","params":{"ids":[0],"effective_at":10}},
            {"id":"graph.set_edge_endpoints","params":{"revisions":[[4,4,0]],"effective_at":10}},
            {"id":"core.init_scalar","params":{"target":"start","value":"0"}},
            {"id":"core.init_scalar","params":{"target":"end","value":6}},
            {"id":"iter.loop","params":{"iterations":2,"body":[shortest,paths,walk]}},
            {"id":"core.attach_node_attr","params":{"source":"distances","attr":"distance","effective_at":20}},
            {"id":"core.broadcast_scalar","params":{"scalar":"paths","reference":"distances","target":"path_rows"}},
            {"id":"core.broadcast_scalar","params":{"scalar":"walk","reference":"distances","target":"walk_rows"}},
            {"id":"core.attach_node_attr","params":{"source":"path_rows","attr":"paths","effective_at":20}},
            {"id":"core.attach_node_attr","params":{"source":"walk_rows","attr":"walk","effective_at":20}}
            ]));
            let plan = p.prepare_durable(&store, &source).unwrap();
            let pending = std::thread::scope(|s| {
                s.spawn(move || plan.execute(&mut Context::new()).unwrap())
                    .join()
                    .unwrap()
            });
            let result = pending.accept(&mut store).unwrap();
            let distances = if weighted {
                [0., 2., 3., 5.]
            } else {
                [0., 1., if directed { 2. } else { 1. }, 1.]
            };
            for (n, d) in [0, 2, 4, 6].into_iter().zip(distances) {
                assert_eq!(
                    result.node_attr(n, &"distance".into()).unwrap(),
                    Some(if weighted {
                        V::Float64(d)
                    } else {
                        V::Int(d as i64)
                    })
                );
            }
            let Some(V::Text(paths)) = result.node_attr(0, &"paths".into()).unwrap() else {
                panic!("paths")
            };
            let paths: Vec<(Vec<usize>, f64)> = serde_json::from_str(&paths).unwrap();
            let expected = match (weighted, directed) {
                (true, true) => vec![(vec![0, 2, 4, 6], 5.), (vec![0, 6], 8.)],
                (true, false) => vec![
                    (vec![0, 2, 4, 6], 5.),
                    (vec![0, 6], 8.),
                    (vec![0, 4, 6], 12.),
                ],
                (false, true) => vec![(vec![0, 6], 1.), (vec![0, 2, 4, 6], 3.)],
                (false, false) => vec![
                    (vec![0, 6], 1.),
                    (vec![0, 4, 6], 2.),
                    (vec![0, 2, 4, 6], 3.),
                ],
            };
            assert_eq!(paths, expected);
            let Some(V::Text(walk)) = result.node_attr(0, &"walk".into()).unwrap() else {
                panic!("walk")
            };
            let walks: Vec<Vec<usize>> = serde_json::from_str(&walk).unwrap();
            assert_eq!(walks.len(), 1);
            assert_eq!(walks[0][0], 0);
            assert!(walks[0].len() <= 7);
            let edges = HashSet::from([(0, 2), (2, 2), (2, 4), (4, 0), (4, 6), (0, 6)]);
            for pair in walks[0].windows(2) {
                assert!(
                    pair[1] == 0
                        || edges.contains(&(pair[0], pair[1]))
                        || (!directed && edges.contains(&(pair[1], pair[0])))
                );
            }
            let owner = graph.into_shared();
            let selected = Subgraph::from_nodes(
                owner.clone(),
                [0, 2, 4].into_iter().collect(),
                "selected".into(),
            )
            .unwrap()
            .freeze()
            .unwrap();
            p.run(&mut Context::new(), selected).unwrap();
            for n in [0, 2, 4, 6] {
                for attr in ["distance", "paths", "walk"] {
                    assert_eq!(
                        owner.borrow().get_node_attr(n, &attr.into()).unwrap(),
                        result.node_attr(n, &attr.into()).unwrap()
                    );
                }
            }
            assert_eq!(
                owner.borrow().get_node_attr(5, &"distance".into()).unwrap(),
                None
            );
            drop(store);
            let recovered = DurableGraph::open(&path).unwrap();
            assert_eq!(
                recovered
                    .read_view_at(19, None)
                    .unwrap()
                    .node_attr(0, &"distance".into())
                    .unwrap(),
                None
            );
            assert_eq!(
                recovered
                    .read_view_at(20, None)
                    .unwrap()
                    .node_attr(6, &"paths".into())
                    .unwrap(),
                result.node_attr(6, &"paths".into()).unwrap()
            );
        }
    }
}
#[test]
fn invalid_path_controls_weights_or_start_domains_preserve_journal_outputs_and_ids() {
    for op in [
        "core.shortest_path_map",
        "core.k_shortest_paths",
        "core.random_walk",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
        let source = store.read_view().unwrap();
        let before = fs::read(&path).unwrap();
        let head = source.write_state_id();
        let params = match op {
            "core.shortest_path_map" => {
                json!({"source":"start","output":"result","weight_attr":"w"})
            }
            "core.k_shortest_paths" => {
                json!({"source":"start","target":"start","k":0,"output":"result","weight_attr":"w"})
            }
            _ => {
                json!({"start_nodes":"start","length":0,"output":"result","weight_attr":"w","seed":19})
            }
        };
        let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"core.init_edges","params":{"value":-1.0,"target":"bad"}},
        {"id":"core.attach_edge_attr","params":{"source":"bad","attr":"w"}},
        {"id":"core.init_scalar","params":{"target":"start","value":0}},
        {"id":op,"params":params}
        ]));
        assert!(p
            .run_durable(&mut Context::new(), &mut store, &source)
            .is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(store.read_view().unwrap().write_state_id(), head);
    }
}
