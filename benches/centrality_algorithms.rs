use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use groggy::algorithms::centrality::{BetweennessCentrality, ClosenessCentrality};
use groggy::algorithms::{Algorithm, Context};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::types::NodeId;

fn build_chain(length: usize) -> Subgraph {
    let mut graph = Graph::new();
    let mut nodes: Vec<NodeId> = Vec::with_capacity(length);
    for _ in 0..length {
        nodes.push(graph.add_node());
    }
    for window in nodes.windows(2) {
        let u = window[0];
        let v = window[1];
        graph.add_edge(u, v).unwrap();
        graph.add_edge(v, u).unwrap();
    }
    let set: HashSet<NodeId> = nodes.into_iter().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), set, "centrality".into()).unwrap()
}

fn bench_betweenness(c: &mut Criterion) {
    let mut group = c.benchmark_group("centrality_betweenness");
    group.bench_function("betweenness_chain_64", |b| {
        b.iter_batched(
            || build_chain(64),
            |subgraph| {
                let algo = BetweennessCentrality::new(true, None, "bc".into());
                let mut ctx = Context::new();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_closeness(c: &mut Criterion) {
    let mut group = c.benchmark_group("centrality_closeness");
    group.bench_function("closeness_chain_64", |b| {
        b.iter_batched(
            || build_chain(64),
            |subgraph| {
                let algo = ClosenessCentrality::new(true, None, "cc".into());
                let mut ctx = Context::new();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(centrality, bench_betweenness, bench_closeness);
criterion_main!(centrality);
