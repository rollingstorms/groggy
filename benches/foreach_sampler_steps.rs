//! Bounded native release receipt for nested selector/property/output frames.
use groggy::{
    algorithms::{
        builder::{register_algorithms, StepSamplePipeline},
        AlgorithmOutput, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context, Registry,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for mode in [
            "flat_selectors",
            "empty_selectors",
            "persistent_overlap",
            "transient_overlap",
            "item_histograms",
            "nested_properties",
        ] {
            let sampler = mode.ends_with("selectors");
            let persist = mode == "persistent_overlap" || mode == "nested_properties";
            let mut body = match mode {
                "flat_selectors" => vec![
                    json!({"id":"sample.iterate_nodes","params":{"target":"nodes"}}),
                    json!({"id":"sample.emit_subgraphs","params":{"source":"nodes","target":"__sample_output__","induced":false}}),
                ],
                "empty_selectors" => vec![
                    json!({"id":"core.sample_nodes","params":{"target":"mask","count":0}}),
                    json!({"id":"sample.emit_subgraphs","params":{"source":"mask","target":"__sample_output__","induced":false}}),
                ],
                "item_histograms" => vec![
                    json!({"id":"core.load_node_attr","params":{"attr":"score","target":"values"}}),
                    json!({"id":"core.histogram","params":{"source":"values","target":"hist","bins":2}}),
                ],
                _ => vec![
                    json!({"id":"core.load_node_attr","params":{"attr":"score","target":"values"}}),
                    json!({"id":"core.map_nodes","params":{"source":"values","target":"next","expr":{"type":"binary_op","op":"add","left":{"type":"var","name":"value"},"right":{"type":"const","value":1}}}}),
                    json!({"id":"core.attach_node_attr","params":{"source":"next","attr":"score"}}),
                ],
            };
            if mode == "nested_properties" {
                body = vec![
                    json!({"id":"sample.iterate_nodes","params":{"target":"nodes"}}),
                    json!({"id":"sample.for_each","params":{"source":"nodes","target":"inner","body":body}}),
                ];
            }
            let mut steps = vec![
                json!({"id":"sample.iterate_edges","params":{"target":"seeds"}}),
                json!({"id":"sample.for_each","params":{"source":"seeds","target":"mapped","body":body}}),
            ];
            if sampler {
                steps.push(json!({"id":"sample.emit_subgraphs","params":{"source":"mapped","target":"__sample_output__","induced":false}}));
            }
            let mut params = AlgorithmParams::new();
            params.insert(
                "steps",
                AlgorithmParamValue::Json(json!({"atomic":!sampler,"steps":steps})),
            );
            let spec = AlgorithmSpec {
                id: if sampler {
                    "builder.sample_pipeline"
                } else {
                    "builder.step_pipeline"
                }
                .into(),
                params,
            };
            let selector_pipeline = if sampler {
                Some(StepSamplePipeline::try_from_spec(&spec)?)
            } else {
                None
            };
            let algorithm = if sampler {
                None
            } else {
                Some(registry.instantiate(&spec)?)
            };
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_directed();
                graph.add_nodes(count);
                for node in 0..count {
                    graph.add_edge(node, (node + 1) % count)?;
                    graph.set_node_attr(node, "score".into(), AttrValue::Int(0))?;
                }
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selected =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "ring".into())?;
                let mut ctx = Context::new();
                ctx.set_persist_results(persist);
                let start = Instant::now();
                let selectors = if let Some(pipeline) = &selector_pipeline {
                    Some(pipeline.run(&mut ctx, selected)?)
                } else {
                    algorithm
                        .as_ref()
                        .ok_or("missing algorithm")?
                        .execute(&mut ctx, selected)?;
                    None
                };
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + u64::from(persist)
                    || graph.has_uncommitted_changes()
                {
                    return Err("invalid publication".into());
                }
                if let Some(selectors) = selectors {
                    if selectors.len()
                        != if mode == "flat_selectors" {
                            2 * count
                        } else {
                            0
                        }
                    {
                        return Err("invalid flattened count".into());
                    }
                    for (index, selector) in selectors.iter().enumerate() {
                        let a = index / 2;
                        let b = (a + 1) % count;
                        let expected = if index % 2 == 0 { a.min(b) } else { a.max(b) };
                        if selector.nodes().len() != 1
                            || !selector.nodes().contains(&expected)
                            || !selector.edges().is_empty()
                        {
                            return Err("invalid ordered selector".into());
                        }
                    }
                } else if persist {
                    for node in 0..count {
                        if !matches!(
                            graph.get_node_attr(node, &"score".into())?,
                            Some(AttrValue::Int(2) | AttrValue::SmallInt(2))
                        ) {
                            return Err("invalid staged overlap".into());
                        }
                    }
                } else {
                    let outputs = ctx.take_outputs();
                    if mode == "item_histograms" {
                        if outputs.len() != count {
                            return Err("lost item histograms".into());
                        }
                        for index in 0..count {
                            if !matches!(outputs.get(&format!("for_each.mapped.{index}.histograms.hist")),Some(AlgorithmOutput::Histogram(rows)) if rows==&vec![2,0])
                            {
                                return Err("invalid item histogram".into());
                            }
                        }
                    } else if !matches!(outputs.get("node_attributes.score"),Some(AlgorithmOutput::NodeAttributes(rows)) if rows.len()==count && rows.iter().all(|(_,v)|v==&AttrValue::Int(1)))
                    {
                        return Err("invalid owned column".into());
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_FOREACH_PROFILE")
        .unwrap_or_else(|_| "foreach_sampler_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release directed rings; six nested selector/property/output frame modes. Fixture/spec construction and validation excluded; five observations per workload. Validate full output cardinality/order, merged columns, per-item histograms and zero/one publication. Legacy connected selectors. No matched speedup, FFI/RSS/allocation or production-scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote foreach receipt to {path}");
    Ok(())
}
