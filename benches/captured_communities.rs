//! Compare shared community kernels without changing legacy preparation or publishing results.
use groggy::algorithms::{community::*, Algorithm, Context};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, Graph, GraphView};
use serde_json::json;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn read(view: &GraphView, kind: usize) -> groggy::GraphResult<usize> {
    Ok(match kind {
        0 => view.louvain(20, 10, 1.0)?.len(),
        1 => view.leiden(20, 10, 1.0, Some(19), 0.01)?.len(),
        2 => view.infomap(0.15, 3, 30, 10000, Some(19))?.len(),
        3 => view.girvan_newman(Some(4), None, None)?.len(),
        _ => view.label_propagation(20, 0.0, None, Some(19))?.len(),
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for kind in 0..5 {
        for count in if kind == 3 {
            vec![40, 100]
        } else {
            vec![300, 3000]
        } {
            let mut graph = Graph::new_manual();
            graph.add_nodes(count)?;
            // Small cliques connected by one bridge, plus an isolated node.
            for base in (0..count - 1).step_by(5) {
                for u in base..(base + 5).min(count - 1) {
                    for v in u + 1..(base + 5).min(count - 1) {
                        graph.add_edge(u, v)?;
                    }
                }
                if base + 5 < count - 1 {
                    graph.add_edge(base + 4, base + 5)?;
                }
            }
            let nodes = (0..count).collect::<HashSet<_>>();
            let edges = graph.read_view()?.edge_ids().into_iter().collect();
            let start = Instant::now();
            let view = graph.read_view()?;
            let capture = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let input = view.algorithm_input()?;
            let preparation = start.elapsed().as_secs_f64();
            let graph = Arc::new(OwnerCell::new(graph));
            let selected = Subgraph::new(graph, nodes, edges, "benchmark".into());
            let algorithm: Box<dyn Algorithm> = match kind {
                0 => Box::new(Louvain::new(20, 10, 1.0, "out".into())?),
                1 => Box::new(Leiden::new(20, 10, 1.0, Some(19), "out".into())?),
                2 => Box::new(Infomap::new(0.15, 3, 30, Some(19), "out".into())?),
                3 => Box::new(GirvanNewman::new(Some(4), None, None, "out".into())),
                _ => Box::new(
                    LabelPropagation::new(20, 0.0, "out".into(), None)?.with_seed(Some(19)),
                ),
            };
            let mut convenience = Vec::new();
            let mut connected = Vec::new();
            let mut prepared = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                assert_eq!(read(&view, kind)?, count);
                convenience.push(start.elapsed().as_secs_f64());
                let mut ctx = Context::new();
                ctx.set_persist_results(false);
                let start = Instant::now();
                black_box(algorithm.execute(&mut ctx, selected.clone())?);
                connected.push(start.elapsed().as_secs_f64());
                let mut ctx = Context::new();
                let start = Instant::now();
                let len = match kind {
                    0 => input
                        .louvain(&Louvain::new(20, 10, 1.0, "out".into())?, &mut ctx)?
                        .len(),
                    1 => input
                        .leiden(&Leiden::new(20, 10, 1.0, Some(19), "out".into())?, &mut ctx)?
                        .len(),
                    2 => input
                        .infomap(
                            &Infomap::new(0.15, 3, 30, Some(19), "out".into())?,
                            &mut ctx,
                        )?
                        .len(),
                    3 => input
                        .girvan_newman(
                            &GirvanNewman::new(Some(4), None, None, "out".into()),
                            &mut ctx,
                        )?
                        .len(),
                    _ => input
                        .label_propagation(
                            &LabelPropagation::new(20, 0.0, "out".into(), None)?
                                .with_seed(Some(19)),
                            &mut ctx,
                        )?
                        .len(),
                };
                assert_eq!(len, count);
                prepared.push(start.elapsed().as_secs_f64());
            }
            cases.push(json!({"algorithm":algorithm.id(),"nodes":count,"edges":view.edge_count(),"capture_seconds":capture,"prepare_seconds":preparation,"connected_seconds":connected,"view_seconds":convenience,"prepared_seconds":prepared}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"five-node cliques with bridges and one isolate; no properties/history; five interleaved samples; no publication","cases":cases})
        )?
    );
    Ok(())
}
