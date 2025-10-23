use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use groggy::algorithms::{ensure_algorithms_registered, PipelineBuilder};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::types::{AttrName, AttrValue, NodeId};

fn toy_graph() -> Subgraph {
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let c = graph.add_node();
    let d = graph.add_node();

    graph.add_edge(a, b).unwrap();
    graph.add_edge(b, a).unwrap();
    graph.add_edge(b, c).unwrap();
    graph.add_edge(c, b).unwrap();
    graph.add_edge(c, d).unwrap();
    graph.add_edge(d, c).unwrap();

    let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap()
}

#[test]
fn pipeline_runs_pagerank() {
    ensure_algorithms_registered();
    let subgraph = toy_graph();
    let builder = PipelineBuilder::new().with_algorithm("centrality.pagerank", |params| {
        params.set_int("max_iter", 25);
        params.set_text("output_attr", "pr");
    });
    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let attr_name: AttrName = "pr".to_string();
    let node = *result.nodes().iter().next().unwrap();
    let attr = result.get_node_attribute(node, &attr_name).unwrap();
    assert!(attr.is_some());
}

#[test]
fn pipeline_runs_betweenness_and_closeness() {
    ensure_algorithms_registered();
    let subgraph = toy_graph();
    let builder = PipelineBuilder::new()
        .with_algorithm("centrality.betweenness", |params| {
            params.set_text("output_attr", "bc");
        })
        .with_algorithm("centrality.closeness", |params| {
            params.set_text("output_attr", "cc");
        });

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let node = *result.nodes().iter().next().unwrap();
    let bc = result
        .get_node_attribute(node, &"bc".into())
        .unwrap()
        .unwrap();
    let cc = result
        .get_node_attribute(node, &"cc".into())
        .unwrap()
        .unwrap();

    match (bc, cc) {
        (AttrValue::Float(_), AttrValue::Float(_)) => {}
        _ => panic!("expected float attributes"),
    }
}

#[test]
fn pipeline_runs_weighted_centrality() {
    ensure_algorithms_registered();
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let c = graph.add_node();

    let ab = graph.add_edge(a, b).unwrap();
    let ba = graph.add_edge(b, a).unwrap();
    let bc = graph.add_edge(b, c).unwrap();
    let cb = graph.add_edge(c, b).unwrap();

    graph
        .set_edge_attr(ab, "weight".into(), AttrValue::Float(1.0))
        .unwrap();
    graph
        .set_edge_attr(ba, "weight".into(), AttrValue::Float(1.0))
        .unwrap();
    graph
        .set_edge_attr(bc, "weight".into(), AttrValue::Float(2.0))
        .unwrap();
    graph
        .set_edge_attr(cb, "weight".into(), AttrValue::Float(2.0))
        .unwrap();

    let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
    let subgraph =
        Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "weighted".into()).unwrap();

    let builder = PipelineBuilder::new()
        .with_algorithm("centrality.betweenness", |params| {
            params.set_text("weight_attr", "weight");
            params.set_text("output_attr", "betweenness_w");
        })
        .with_algorithm("centrality.closeness", |params| {
            params.set_text("weight_attr", "weight");
            params.set_text("output_attr", "closeness_w");
        });

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let bw = result
        .get_node_attribute(b, &"betweenness_w".into())
        .unwrap()
        .unwrap();
    let cw = result
        .get_node_attribute(b, &"closeness_w".into())
        .unwrap()
        .unwrap();

    match (bw, cw) {
        (AttrValue::Float(_), AttrValue::Float(_)) => {}
        _ => panic!("expected float attributes"),
    }
}
