use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use groggy::algorithms::{ensure_algorithms_registered, PipelineBuilder};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::types::NodeId;

fn sample_graph() -> Subgraph {
    let mut graph = Graph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let c = graph.add_node();
    let d = graph.add_node();

    graph.add_edge(a, b).unwrap();
    graph.add_edge(b, a).unwrap();
    graph.add_edge(c, d).unwrap();
    graph.add_edge(d, c).unwrap();
    graph.add_edge(b, c).unwrap();
    graph.add_edge(c, b).unwrap();

    let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap()
}

#[test]
fn pipeline_runs_label_propagation() {
    ensure_algorithms_registered();
    let subgraph = sample_graph();
    let builder = PipelineBuilder::new().with_algorithm("community.lpa", |params| {
        params.set_int("max_iter", 15);
        params.set_text("output_attr", "lpa_comm");
    });
    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let first_node = *result.nodes().iter().next().unwrap();
    let attr_a = result
        .get_node_attribute(first_node, &"lpa_comm".into())
        .unwrap();
    assert!(attr_a.is_some());
}

#[test]
fn pipeline_runs_louvain() {
    ensure_algorithms_registered();
    let subgraph = sample_graph();
    let builder = PipelineBuilder::new().with_algorithm("community.louvain", |params| {
        params.set_int("max_iter", 10);
        params.set_text("output_attr", "louvain_comm");
    });
    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .unwrap();
    let mut ctx = groggy::algorithms::Context::new();
    let result = pipeline.run(&mut ctx, subgraph).unwrap();

    let first_node = *result.nodes().iter().next().unwrap();
    let attr = result
        .get_node_attribute(first_node, &"louvain_comm".into())
        .unwrap();
    assert!(attr.is_some());
}
