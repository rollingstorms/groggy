use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use groggy::algorithms::pathfinding::{BfsTraversal, DfsTraversal, DijkstraShortestPath};
use groggy::algorithms::{Algorithm, Context};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::types::{AttrValue, NodeId};

fn make_line_graph(length: usize, start_attr: &str) -> Subgraph {
    let mut graph = Graph::new();
    let mut nodes: Vec<NodeId> = Vec::with_capacity(length);
    for _ in 0..length {
        nodes.push(graph.add_node());
    }
    if let Some(first) = nodes.first() {
        graph
            .set_node_attr(*first, start_attr.into(), AttrValue::Bool(true))
            .unwrap();
    }
    for window in nodes.windows(2) {
        let u = window[0];
        let v = window[1];
        graph.add_edge(u, v).unwrap();
        graph.add_edge(v, u).unwrap();
    }
    let set: HashSet<NodeId> = nodes.into_iter().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), set, "path".into()).unwrap()
}

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathfinding_bfs");
    group.bench_function("bfs_chain_512", |b| {
        b.iter_batched(
            || make_line_graph(512, "start"),
            |subgraph| {
                let algo = BfsTraversal::new("start".into(), "distance".into());
                let mut ctx = Context::new();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathfinding_dijkstra");
    group.bench_function("dijkstra_chain_256", |b| {
        b.iter_batched(
            || make_line_graph(256, "start"),
            |subgraph| {
                let algo = DijkstraShortestPath::new("start".into(), None, "distance".into());
                let mut ctx = Context::new();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathfinding_dfs");
    group.bench_function("dfs_chain_512", |b| {
        b.iter_batched(
            || make_line_graph(512, "start"),
            |subgraph| {
                let algo = DfsTraversal::new("start".into(), "order".into());
                let mut ctx = Context::new();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(pathfinding, bench_bfs, bench_dijkstra, bench_dfs);
criterion_main!(pathfinding);
