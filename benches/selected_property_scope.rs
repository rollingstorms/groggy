//! Bounded native property-read cost with selected membership versus owner access.
use groggy::{
    subgraphs::Subgraph, traits::SubgraphOperations, AttrValue as V, BatchNodeRef as N, Graph,
    GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 10_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count + 1,
                node_attrs: [(
                    "score".into(),
                    (0..=count)
                        .map(|i| (N::Added(i), V::Int(i as i64)))
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let owner = graph.into_shared();
        let selected = Subgraph::new(
            owner.clone(),
            (0..count).collect(),
            Default::default(),
            "selected".into(),
        );
        let name = "score".to_string();
        let repetitions = 100_000 / count;
        let expected = (count as u64 * (count - 1) as u64 / 2) * repetitions as u64;
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let mut times = [0.; 2];
            for kind in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                let mut checksum = 0u64;
                for _ in 0..repetitions {
                    for id in 0..count {
                        let value = if kind == 0 {
                            selected.get_node_attribute(id, &name)?
                        } else {
                            owner.borrow().get_node_attr(id, &name)?
                        };
                        checksum += match value {
                            Some(V::Int(value)) => value as u64,
                            Some(V::SmallInt(value)) => value as u64,
                            _ => return Err("Unexpected fixture value".into()),
                        };
                    }
                }
                times[kind] = start.elapsed().as_secs_f64() / (repetitions * count) as f64;
                assert_eq!(checksum, expected);
                black_box(checksum);
            }
            samples.push(json!({"selected_read_seconds":times[0],"owner_read_seconds":times[1]}));
        }
        assert!(selected.get_node_attribute(count, &name).is_err());
        cases.push(
            json!({"selected_nodes":count,"reads_per_sample":repetitions*count,"samples":samples}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"sequential scalar reads; selected graph excludes one otherwise active owner node","cases":cases})
        )?
    );
    Ok(())
}
