//! Matched pending capture workload with zero, half or all large columns replaced.
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmOutput, AlgorithmParams, AlgorithmSpec, Context,
        Registry,
    },
    subgraphs::Subgraph,
    AttrValue, BatchEdgeRef, BatchNodeRef, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::time::Instant;

pub fn run(
    mut start: impl FnMut(),
    mut stop: impl FnMut() -> Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for nodes in [true, false] {
        for count in [100, 1_000] {
            for width in [8, 1_024] {
                for replacements in [0, 1, 2] {
                    let mut steps = Vec::new();
                    if replacements != 0 {
                        steps.push(json!({"id":if nodes {"core.init_nodes"} else {"core.init_edges"},"params":{"target":"replacement","value":7}}));
                        for column in 0..replacements {
                            steps.push(json!({"id":if nodes {"core.attach_node_attr"} else {"core.attach_edge_attr"},"params":{"source":"replacement","attr":format!("payload{column}")}}));
                        }
                    }
                    steps.push(json!({"id":"graph.capture_staged","params":{"output":"captured"}}));
                    steps.push(
                        json!({"id":"graph.emit","params":{"source":"captured","name":"captured"}}),
                    );
                    let mut params = AlgorithmParams::new();
                    params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
                    let algorithm = registry.instantiate(&AlgorithmSpec {
                        id: "builder.step_pipeline".into(),
                        params,
                    })?;
                    let mut samples = Vec::new();
                    for _ in 0..5 {
                        let values = (0..width).map(|i| i as f64 + 0.125).collect::<Vec<_>>();
                        let mut batch = GraphBatch {
                            add_nodes: if nodes { count } else { count + 1 },
                            ..Default::default()
                        };
                        if !nodes {
                            batch.add_edges = (0..count)
                                .map(|id| (BatchNodeRef::Added(id), BatchNodeRef::Added(id + 1)))
                                .collect();
                        }
                        for column in 0..2 {
                            let name = format!("payload{column}");
                            if nodes {
                                batch.node_attrs.insert(
                                    name,
                                    (0..count)
                                        .map(|id| {
                                            (
                                                BatchNodeRef::Added(id),
                                                AttrValue::Float64Vec(values.clone()),
                                            )
                                        })
                                        .collect(),
                                );
                            } else {
                                batch.edge_attrs.insert(
                                    name,
                                    (0..count)
                                        .map(|id| {
                                            (
                                                BatchEdgeRef::Added(id),
                                                AttrValue::Float64Vec(values.clone()),
                                            )
                                        })
                                        .collect(),
                                );
                            }
                        }
                        let mut graph = Graph::new();
                        graph.apply_batch(batch, 0, "fixture".into(), "bench".into())?;
                        let owner = graph.into_shared();
                        let before = owner.borrow().read_view()?;
                        let selected = Subgraph::from_frozen_view(
                            owner.clone(),
                            before.clone(),
                            "overwrite benchmark".into(),
                        );
                        let mut ctx = Context::new();
                        start();
                        let timer = Instant::now();
                        let result = algorithm.execute(&mut ctx, selected)?;
                        let seconds = timer.elapsed().as_secs_f64();
                        let allocations = stop();
                        let outputs = ctx.take_outputs();
                        let AlgorithmOutput::GraphView(captured) = &outputs["captured"] else {
                            return Err("missing capture".into());
                        };
                        assert_eq!(
                            owner.borrow().current_state_id(),
                            if replacements == 0 { 1 } else { 2 }
                        );
                        assert_eq!(result.node_count(), if nodes { count } else { count + 1 });
                        assert_eq!(result.edge_count(), if nodes { 0 } else { count });
                        for id in 0..count {
                            for column in 0..2 {
                                let name = format!("payload{column}");
                                let old = if nodes {
                                    before.node_attr(id, &name)?
                                } else {
                                    before.edge_attr(id, &name)?
                                };
                                assert_eq!(old, Some(AttrValue::Float64Vec(values.clone())));
                                let current = if nodes {
                                    captured.node_attr(id, &name)?
                                } else {
                                    captured.edge_attr(id, &name)?
                                };
                                if column < replacements {
                                    assert_eq!(
                                        current.as_ref().and_then(AttrValue::as_int),
                                        Some(7)
                                    );
                                } else {
                                    assert_eq!(current, old);
                                }
                            }
                        }
                        samples.push(json!({"seconds":seconds,"allocations":allocations}));
                    }
                    cases.push(json!({"entity":if nodes {"node"} else {"edge"},"entities":count,"vector_width":width,"replaced_columns":replacements,"samples":samples}));
                }
            }
        }
    }
    Ok(
        json!({"scope":"Matched native release pending graph capture and zero/one/two scalar replacements of two f64 vector columns. Five observations per case. Includes effect staging, selected immutable capture, emission and zero/one publication; excludes fixture, initial accepted pin, algorithm construction, assertions, destruction and compilation. Every entity/value and immutable prior input is checked. Compare identical cases before/after in fresh processes; allocator requested bytes are not RSS or a production scale budget.","cases":cases}),
    )
}
