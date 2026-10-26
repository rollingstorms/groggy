//! Registry algorithms compose through the same outer property transaction as builders.
use groggy::algorithms::{global_registry, pipeline::PipelineBuilder, AlgorithmParams, Context};
use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

fn fixture() -> (Rc<RefCell<Graph>>, Subgraph) {
    let mut graph = Graph::new();
    graph.add_nodes(4).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph.add_edge(1, 2).unwrap();
    graph.add_edge(0, 1).unwrap(); // parallel route
    graph.add_edge(2, 3).unwrap(); // deliberately outside the selection
    graph
        .set_node_attr(0, "start".into(), AttrValue::Bool(false))
        .unwrap();
    graph
        .set_node_attr(2, "goal".into(), AttrValue::Null)
        .unwrap();
    let graph = Rc::new(RefCell::new(graph));
    let selected = Subgraph::new(
        graph.clone(),
        HashSet::from([0, 1, 2]),
        HashSet::from([0, 1, 2]),
        "selected".into(),
    );
    (graph, selected)
}
fn definition(params: &mut AlgorithmParams, steps: serde_json::Value) {
    params.set_text(
        "steps",
        serde_json::json!({"steps":steps,"atomic":true}).to_string(),
    );
}
fn assign(params: &mut AlgorithmParams, name: &str, value: i64, edge: bool) {
    let (init, attach) = if edge {
        ("core.init_edges", "core.attach_edge_attr")
    } else {
        ("core.init_nodes", "core.attach_node_attr")
    };
    definition(
        params,
        serde_json::json!([
            {"id":init,"params":{"target":"value","value":value}},
            {"id":attach,"params":{"source":"value","attr":name}}
        ]),
    );
}
fn configure(params: &mut AlgorithmParams) {
    params.set_text("output_attr", "result");
    params.set_text("start_attr", "start");
    params.set_text("goal_attr", "goal");
    params.set_int("max_iter", 10);
    params.set_int("num_levels", 2);
}
const CLASSIC: &[&str] = &[
    "centrality.pagerank",
    "centrality.closeness",
    "centrality.betweenness",
    "community.louvain",
    "community.leiden",
    "community.girvan_newman",
    "community.lpa",
    "community.connected_components",
    "community.infomap",
    "pathfinding.astar",
    "pathfinding.bfs",
    "pathfinding.dfs",
    "pathfinding.dijkstra",
];

#[test]
fn all_classic_registry_algorithms_publish_once_and_feed_later_builder_reads() {
    for id in CLASSIC {
        let (graph, selected) = fixture();
        let before = graph.borrow().current_state_id();
        let frozen = graph.borrow().read_view().unwrap();
        let pipe = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |p| assign(p, "prefix", 7, false))
            .with_algorithm(*id, configure)
            .with_algorithm("builder.step_pipeline", |p| {
                definition(
                    p,
                    serde_json::json!([
                        {"id":"core.load_node_attr","params":{"target":"loaded","attr":"result"}},
                        {"id":"core.attach_node_attr","params":{"source":"loaded","attr":"copy"}}
                    ]),
                )
            })
            .build(global_registry())
            .unwrap();
        pipe.run_atomic_properties(&mut Context::new(), selected)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        let g = graph.borrow();
        assert_eq!(g.current_state_id(), before + 1, "{id}");
        assert!(!g.has_uncommitted_changes(), "{id}");
        for node in 0..3 {
            let value = g.get_node_attr(node, &"result".into()).unwrap();
            assert!(value.is_some(), "{id}: missing {node}");
            assert_eq!(
                value,
                g.get_node_attr(node, &"copy".into()).unwrap(),
                "{id}"
            );
        }
        assert_eq!(g.get_node_attr(3, &"result".into()).unwrap(), None, "{id}");
        assert_eq!(frozen.node_attr(0, &"result".into()).unwrap(), None);
    }
}

#[test]
fn later_managed_property_failure_rolls_back_every_classic_family_and_reuses_context() {
    for id in CLASSIC {
        let (graph, selected) = fixture();
        let before = graph.borrow().current_state_id();
        let pipe = PipelineBuilder::new()
            .with_algorithm(*id, configure)
            .with_algorithm("builder.step_pipeline", |p| {
                assign(p, "entity_type", 3, false)
            })
            .build(global_registry())
            .unwrap();
        let mut ctx = Context::new();
        assert!(
            pipe.run_atomic_properties(&mut ctx, selected.clone())
                .is_err(),
            "{id}"
        );
        assert_eq!(graph.borrow().current_state_id(), before, "{id}");
        assert_eq!(
            graph.borrow().get_node_attr(0, &"result".into()).unwrap(),
            None,
            "{id}"
        );
        let good = PipelineBuilder::new()
            .with_algorithm(*id, configure)
            .build(global_registry())
            .unwrap();
        good.run_atomic_properties(&mut ctx, selected).unwrap();
        assert_eq!(graph.borrow().current_state_id(), before + 1, "{id}");
    }
}

#[test]
fn weighted_algorithms_validate_staged_edges_and_discard_prefixes() {
    for id in [
        "centrality.closeness",
        "centrality.betweenness",
        "community.girvan_newman",
        "pathfinding.astar",
        "pathfinding.dijkstra",
    ] {
        let (graph, selected) = fixture();
        let before = graph.borrow().current_state_id();
        let pipe = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |p| assign(p, "length", -1, true))
            .with_algorithm(id, |p| {
                configure(p);
                p.set_text("weight_attr", "length");
            })
            .build(global_registry())
            .unwrap();
        let err = pipe
            .run_atomic_properties(&mut Context::new(), selected)
            .unwrap_err();
        assert!(err.to_string().contains("nonnegative"), "{id}: {err}");
        assert_eq!(graph.borrow().current_state_id(), before, "{id}");
        assert_eq!(
            graph.borrow().get_edge_attr(0, &"length".into()).unwrap(),
            None,
            "{id}"
        );
    }
}

#[test]
fn staged_markers_personalization_heuristics_and_label_seeds_are_visible() {
    for (id, name, param, error) in [
        ("pathfinding.dijkstra", "mark", "start_attr", "exactly one"),
        ("pathfinding.bfs", "mark", "start_attr", "exactly one"),
        ("pathfinding.dfs", "mark", "start_attr", "exactly one"),
        ("pathfinding.astar", "mark", "start_attr", "multiple nodes"),
        ("pathfinding.astar", "hint", "heuristic_attr", "nonnegative"),
        (
            "centrality.pagerank",
            "personal",
            "personalization_attr",
            "nonnegative",
        ),
    ] {
        let (graph, selected) = fixture();
        let before = graph.borrow().current_state_id();
        let pipe = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |p| assign(p, name, -1, false))
            .with_algorithm(id, |p| {
                configure(p);
                p.set_text(param, name);
            })
            .build(global_registry())
            .unwrap();
        let failure = pipe
            .run_atomic_properties(&mut Context::new(), selected)
            .unwrap_err();
        assert!(
            failure.to_string().contains(error),
            "{id}/{param}: {failure}"
        );
        assert_eq!(graph.borrow().current_state_id(), before);
    }
    let (graph, selected) = fixture();
    let pipe = PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| assign(p, "seed", 8, false))
        .with_algorithm("community.lpa", |p| {
            configure(p);
            p.set_text("seed_attr", "seed");
        })
        .build(global_registry())
        .unwrap();
    pipe.run_atomic_properties(&mut Context::new(), selected)
        .unwrap();
    for n in 0..3 {
        assert_eq!(
            graph.borrow().get_node_attr(n, &"result".into()).unwrap(),
            Some(AttrValue::Int(8))
        );
    }
}
