//! Registry algorithms compose through the same outer property transaction as builders.
use groggy::algorithms::{global_registry, pipeline::PipelineBuilder, AlgorithmParams, Context};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue, Graph};
use std::collections::HashSet;
use std::sync::Arc;

fn fixture() -> (Arc<OwnerCell<Graph>>, Subgraph) {
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
    let graph = Arc::new(OwnerCell::new(graph));
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

#[test]
fn classic_paths_consume_pending_topology_and_leave_one_live_continuation() {
    for id in [
        "pathfinding.bfs",
        "pathfinding.dfs",
        "pathfinding.dijkstra",
        "pathfinding.astar",
    ] {
        let (owner, selected) = fixture();
        selected.adjacency_read().unwrap(); // Warm the old live CSR before pending revisions.
        let head = owner.borrow().current_state_id();
        let pipe = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |p| definition(p, serde_json::json!([
                {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
                {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}],[{"added":0},2]],"output":"edges"}},
                {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,2,0]]}},
                {"id":"graph.remove_nodes","params":{"ids":[1]}},
                {"id":"core.init_edges","params":{"value":2,"target":"weights"}},
                {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}}
            ])))
            .with_algorithm(id, |p| {configure(p);p.set_text("weight_attr", "weight");})
            .with_algorithm("builder.step_pipeline", |p| definition(p, serde_json::json!([
                {"id":"core.load_node_attr","params":{"attr":"result","target":"loaded","default":-1}},
                {"id":"core.attach_node_attr","params":{"source":"loaded","attr":"copy"}}
            ])))
            .build(global_registry()).unwrap();
        let result = pipe
            .run_atomic_properties(&mut Context::new(), selected)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(result.node_count(), 3);
        assert_eq!(owner.borrow().current_state_id(), head + 1);
        let graph = owner.borrow();
        assert!(!graph.has_node(1));
        assert!(graph.has_node(4));
        assert_eq!(graph.edge_endpoints(0).unwrap(), (2, 0));
        assert_eq!(graph.get_node_attr(3, &"result".into()).unwrap(), None);
        let expected = if id == "pathfinding.dijkstra" {
            AttrValue::Float64(2.0)
        } else {
            AttrValue::Int(1)
        };
        assert_eq!(
            graph.get_node_attr(2, &"result".into()).unwrap(),
            Some(expected.clone())
        );
        assert_eq!(
            graph.get_node_attr(2, &"copy".into()).unwrap(),
            Some(expected)
        );
        assert_eq!(
            graph.get_node_attr(4, &"copy".into()).unwrap(),
            Some(match id {
                "pathfinding.bfs" => AttrValue::Int(1),
                "pathfinding.dfs" => AttrValue::Int(2),
                "pathfinding.dijkstra" => AttrValue::Float64(2.0),
                _ => AttrValue::Int(-1),
            })
        );
    }
}
