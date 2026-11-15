//! Shared native child-frame fixture for ordinary timing and allocation receipts.
use groggy::{
    algorithms::{builder::register_algorithms, AlgorithmParams, AlgorithmSpec, Context, Registry},
    ownership::OwnerCell,
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::{json, Value};
use std::{sync::Arc, time::Instant};

pub fn run(
    mut before: impl FnMut(),
    mut after: impl FnMut() -> Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [100, 1000] {
        for topology in [false, true] {
            let mut steps = Vec::new();
            if topology {
                steps.push(json!({"id":"graph.add_nodes","params":{"count":1,"output":"added"}}));
            }
            steps.push(json!({"id":"sample.iterate_nodes","params":{"target":"seeds"}}));
            steps.push(json!({"id":"sample.for_each","params":{"source":"seeds","target":"mapped","body":[
                {"id":"core.load_node_attr","params":{"attr":"counter","target":"values","default":0}},
                {"id":"core.init_scalar","params":{"value":1,"target":"one"}},
                {"id":"core.add","params":{"left":"values","right":"one","target":"next"}},
                {"id":"core.attach_node_attr","params":{"source":"next","attr":"counter"}}
            ]}}));
            let mut params = AlgorithmParams::new();
            params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            let mut allocations = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                let graph = Arc::new(OwnerCell::new(graph));
                let selected = Subgraph::from_frozen_view(
                    graph.clone(),
                    graph.borrow().read_view()?,
                    "bench".into(),
                );
                let mut ctx = Context::new();
                before();
                let started = Instant::now();
                let result = algorithm.execute(&mut ctx, selected)?;
                samples.push(started.elapsed().as_secs_f64());
                allocations.push(after());
                let expected = count + usize::from(topology);
                assert_eq!(result.node_count(), expected);
                assert_eq!(graph.borrow().current_state_id(), 2);
                for node in 0..expected {
                    assert_eq!(
                        graph.borrow().get_node_attr(node, &"counter".into())?,
                        Some(AttrValue::Int(1))
                    );
                }
            }
            cases
                .push(json!({"initial_nodes":count,"pending_topology":topology,"seconds":samples,"allocations":allocations}));
        }
    }
    Ok(
        json!({"scope":"Native release, five observations each. Node-only graph, optional staged node, immutable singleton production and per-child load/increment/attach, one publication. Includes frame setup, pending materialization, validation and execution; excludes fixture/initial capture, compilation, algorithm construction and assertions. Allocation-mode timing includes counter overhead; compare only matched instrumentation. Requested allocations exclude allocator overhead and RSS. No FFI or general scale guarantee.","cases":cases}),
    )
}
