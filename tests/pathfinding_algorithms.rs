use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use groggy::algorithms::{ensure_algorithms_registered, PipelineBuilder};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::types::{AttrName, AttrValue, NodeId};

fn path_graph() -> Subgraph {
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let c = graph.add_node();

    graph.add_edge(a, b).unwrap();
    graph.add_edge(b, c).unwrap();
    graph.add_edge(c, b).unwrap();

    graph
        .set_node_attr(a, "start".into(), AttrValue::Bool(true))
        .unwrap();
    graph
        .set_node_attr(c, "goal".into(), AttrValue::Bool(true))
        .unwrap();

    let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap()
}

#[test]
fn pipeline_runs_bfs_and_dfs() {
    ensure_algorithms_registered();
    let subgraph = path_graph();
    let builder = PipelineBuilder::new()
        .with_algorithm("pathfinding.bfs", |params| {
            params.set_text("start_attr", "start");
            params.set_text("output_attr", "bfs_distance");
        })
        .with_algorithm("pathfinding.dfs", |params| {
            params.set_text("start_attr", "start");
            params.set_text("output_attr", "dfs_order");
        });

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let attr_name: AttrName = "bfs_distance".to_string();
    let dist = result
        .get_node_attribute(result.nodes().iter().copied().next().unwrap(), &attr_name)
        .unwrap();
    assert!(dist.is_some());
}

#[test]
fn pipeline_runs_dijkstra() {
    ensure_algorithms_registered();
    let subgraph = path_graph();
    let builder = PipelineBuilder::new().with_algorithm("pathfinding.dijkstra", |params| {
        params.set_text("start_attr", "start");
        params.set_text("output_attr", "distance");
    });

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let attr_name: AttrName = "distance".to_string();
    let dist = result
        .get_node_attribute(result.nodes().iter().copied().next().unwrap(), &attr_name)
        .unwrap();
    assert!(dist.is_some());
}

#[test]
fn pipeline_runs_astar() {
    ensure_algorithms_registered();
    let subgraph = path_graph();
    let builder = PipelineBuilder::new().with_algorithm("pathfinding.astar", |params| {
        params.set_text("start_attr", "start");
        params.set_text("goal_attr", "goal");
        params.set_text("output_attr", "path_index");
    });

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let attr_name: AttrName = "path_index".to_string();
    let start_node = result
        .nodes()
        .iter()
        .copied()
        .find(|&node| {
            result
                .get_node_attribute(node, &"start".into())
                .ok()
                .flatten()
                .is_some()
        })
        .unwrap();
    let path_idx = result.get_node_attribute(start_node, &attr_name).unwrap();
    assert!(path_idx.is_some());
}
