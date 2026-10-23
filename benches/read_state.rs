//! Bounded current-state capture versus shared read-pin clone scaling.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use groggy::{AttrValue, Graph};
use std::time::Duration;
fn read_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_state");
    for count in [1_000, 10_000] {
        let mut graph = Graph::new();
        let nodes = graph.add_nodes(count);
        for (index, &node) in nodes.iter().enumerate() {
            graph.add_edge(node, nodes[(index + 1) % count]).unwrap();
        }
        graph
            .set_node_attrs(std::collections::HashMap::from([(
                "value".into(),
                nodes
                    .iter()
                    .map(|&id| (id, AttrValue::Int(id as i64)))
                    .collect(),
            )]))
            .unwrap();
        let pin = graph.pin_current_state().unwrap();
        group.bench_with_input(BenchmarkId::new("capture", count), &graph, |b, graph| {
            b.iter(|| black_box(graph.pin_current_state().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("clone", count), &pin, |b, pin| {
            b.iter(|| black_box(pin.clone()));
        });
        group.bench_with_input(BenchmarkId::new("full_view", count), &pin, |b, pin| {
            b.iter(|| black_box(groggy::GraphView::all(pin.clone())));
        });
        let selected = groggy::GraphView::all(pin.clone())
            .select_nodes(&nodes[..count / 2])
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("selected_clone", count),
            &selected,
            |b, view| {
                b.iter(|| black_box(view.clone()));
            },
        );
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1));targets=read_state}
criterion_main!(benches);
