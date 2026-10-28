//! Bounded reverse-reference lookup versus an independent selected-column scan.
use groggy::{
    query::GroupReference, AttrValue as V, BatchNodeRef as N, Graph, GraphBatch, GraphResult,
    GraphView,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn scan(view: &GraphView, group: usize) -> GraphResult<Vec<GroupReference>> {
    let mut rows = Vec::new();
    let attributes = &view.read_state().snapshot().node_attributes;
    for &id in view.nodes() {
        if attributes.get(&id).and_then(|row| row.get("evidence")) == Some(&V::SubgraphRef(group)) {
            rows.push(GroupReference {
                entity_id: id,
                property: "evidence".into(),
            });
        }
    }
    rows.sort_unstable();
    Ok(rows)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [100usize, 10_000] {
        let mut source = Graph::new();
        source.add_node()?;
        let mut parent = Graph::new();
        let child = parent.retain_group(&source.read_view()?, "leaf".into())?;
        parent.create_meta_node(child)?;
        let mut graph = Graph::new();
        let outer = graph.retain_group(&parent.read_view()?, "parent".into())?;
        graph.retain_group(&source.read_view()?, "leaf".into())?;
        graph.apply_batch(
            GraphBatch {
                add_nodes: count + 1,
                node_attrs: [(
                    "evidence".into(),
                    (0..=count)
                        .map(|i| {
                            (
                                N::Added(i),
                                V::SubgraphRef(if i == count { child } else { outer }),
                            )
                        })
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            graph.current_state_id(),
            "bulk roots".into(),
            "bench".into(),
        )?;
        let view = graph.read_view()?;
        let start = Instant::now();
        let index = view.build_group_index()?;
        let build = start.elapsed().as_secs_f64();
        let expected = scan(&view, child)?;
        assert_eq!(index.node_references(child, false)?, expected);
        assert_eq!(index.node_references(child, true)?.len(), count + 1);
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let mut times = [0.; 2];
            for kind in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..100 {
                    let rows = if kind == 0 {
                        index.node_references(child, false)?
                    } else {
                        scan(black_box(&view), child)?
                    };
                    assert_eq!(rows, expected);
                    black_box(rows);
                }
                times[kind] = start.elapsed().as_secs_f64() / 100.;
            }
            let start = Instant::now();
            let recursive = index.node_references(child, true)?;
            let recursive_seconds = start.elapsed().as_secs_f64();
            assert_eq!(recursive.len(), count + 1);
            black_box(recursive);
            samples.push(json!({"indexed_direct_seconds":times[0],"selected_column_scan_seconds":times[1],"indexed_recursive_seconds":recursive_seconds}));
        }
        cases.push(json!({"root_nodes":count+1,"reachable_groups":index.group_ids().len(),"direct_hits":expected.len(),"recursive_hits":count+1,"build_seconds":build,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"two retained revisions; many roots refer to parent and one directly to nested leaf","cases":cases})
        )?
    );
    Ok(())
}
