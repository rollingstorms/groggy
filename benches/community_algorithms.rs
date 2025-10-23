use std::cell::RefCell;
use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use groggy::algorithms::community::{LabelPropagation, Louvain};
use groggy::algorithms::{Algorithm, Context};
use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::types::NodeId;
use std::collections::HashSet;

fn build_subgraph(num_clusters: usize, cluster_size: usize) -> Subgraph {
    let mut graph = Graph::new();
    let mut clusters: Vec<Vec<NodeId>> = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut nodes = Vec::with_capacity(cluster_size);
        for _ in 0..cluster_size {
            nodes.push(graph.add_node());
        }
        clusters.push(nodes);
    }

    // Connect nodes densely inside each cluster and sparsely across clusters.
    for nodes in &clusters {
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let a = nodes[i];
                let b = nodes[j];
                graph.add_edge(a, b).unwrap();
                graph.add_edge(b, a).unwrap();
            }
        }
    }

    for c in 0..clusters.len() - 1 {
        let a = clusters[c][0];
        let b = clusters[c + 1][0];
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
    }

    let all_nodes: HashSet<NodeId> = clusters.into_iter().flatten().collect();
    Subgraph::from_nodes(Rc::new(RefCell::new(graph)), all_nodes, "bench".into()).unwrap()
}

fn bench_lpa(c: &mut Criterion) {
    let mut group = c.benchmark_group("community_lpa");
    group.bench_function("lpa_4x32", |b| {
        b.iter_batched(
            || build_subgraph(4, 32),
            |subgraph| {
                let mut ctx = Context::new();
                let algo = LabelPropagation::new(25, 0.001, "community".into(), None).unwrap();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("community_louvain");
    group.bench_function("louvain_4x32", |b| {
        b.iter_batched(
            || build_subgraph(4, 32),
            |subgraph| {
                let mut ctx = Context::new();
                let algo = Louvain::new(20, 1, 1.0, "community".into()).unwrap();
                let _ = algo.execute(&mut ctx, subgraph).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(community, bench_lpa, bench_louvain);
criterion_main!(community);
