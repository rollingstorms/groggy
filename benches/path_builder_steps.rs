//! Bounded native checked path builder and publication receipt.
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context,
        Registry,
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
            "shortest_unweighted",
            "shortest_weighted",
            "k_shortest",
            "walk_uniform",
            "walk_weighted",
            "walk_many_starts",
        ] {
            let shortest = mode.starts_with("shortest_");
            let weighted = mode.ends_with("weighted") && !mode.ends_with("unweighted");
            let many = mode == "walk_many_starts";
            let mut steps = vec![
                json!({"id":"core.init_scalar","params":{"target":"source","value":0}}),
                json!({"id":"core.init_scalar","params":{"target":"target","value":3}}),
            ];
            if weighted {
                steps.push(
                    json!({"id":"core.init_edges","params":{"target":"weights","value":2.0}}),
                );
                steps.push(json!({"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}}));
            }
            let mut params = if shortest {
                json!({"source":"source","output":"result"})
            } else if mode == "k_shortest" {
                json!({"source":"source","target":"target","k":10,"output":"scalar"})
            } else {
                json!({"start_nodes":if many{"starts"}else{"source"},"length":if many{8}else{64},"seed":19,"output":"scalar"})
            };
            if weighted {
                params["weight_attr"] = json!("weight");
            }
            if many {
                steps.push(json!({"id":"core.init_nodes","params":{"target":"starts","value":0}}));
            }
            steps.push(json!({"id":if shortest{"core.shortest_path_map"}else if mode=="k_shortest"{"core.k_shortest_paths"}else{"core.random_walk"},"params":params}));
            if !shortest {
                steps.push(json!({"id":"core.init_nodes","params":{"target":"all","value":0}}));
                steps.push(
                    json!({"id":"core.top_k","params":{"source":"all","target":"reference","k":1}}),
                );
                steps.push(json!({"id":"core.broadcast_scalar","params":{"scalar":"scalar","reference":"reference","target":"result"}}));
            }
            steps.push(
                json!({"id":"core.attach_node_attr","params":{"source":"result","attr":"answer"}}),
            );
            let mut params = AlgorithmParams::new();
            params.insert(
                "steps",
                AlgorithmParamValue::Json(json!({"atomic":true,"steps":steps})),
            );
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_directed();
                graph.add_nodes(count);
                for node in 0..count {
                    let edge = graph.add_edge(node, (node + 1) % count)?;
                    graph.set_edge_attr(edge, "weight".into(), AttrValue::Float64(1.0))?;
                }
                let edge = graph.add_edge(0, 2)?;
                graph.set_edge_attr(edge, "weight".into(), AttrValue::Float64(1.0))?;
                graph.commit("fixture".into(), "bench".into())?;
                let before = graph.current_state_id();
                let shared = Rc::new(RefCell::new(graph));
                let selected =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "paths".into())?;
                let start = Instant::now();
                algorithm.execute(&mut Context::new(), selected)?;
                let seconds = start.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != before + 1 || graph.has_uncommitted_changes() {
                    return Err("invalid publication".into());
                }
                if shortest {
                    for node in 0..count {
                        let hops = if node == 0 {
                            0
                        } else if node == 1 {
                            1
                        } else {
                            node - 1
                        };
                        let actual = graph.get_node_attr(node, &"answer".into())?;
                        let expected = if weighted {
                            AttrValue::Float64((2 * hops) as f64)
                        } else {
                            AttrValue::Int(hops as i64)
                        };
                        if actual != Some(expected) {
                            return Err(
                                format!("invalid distance {mode} node={node}: {actual:?}").into()
                            );
                        }
                    }
                } else {
                    let Some(AlgorithmParamValue::Text(value)) = graph
                        .get_node_attr(0, &"answer".into())?
                        .and_then(AlgorithmParamValue::from_attr_value)
                    else {
                        return Err("missing serialized result".into());
                    };
                    if mode == "k_shortest" {
                        let paths: Vec<(Vec<usize>, f64)> = serde_json::from_str(&value)?;
                        if paths != vec![(vec![0, 2, 3], 2.0), (vec![0, 1, 2, 3], 3.0)] {
                            return Err("invalid node paths".into());
                        }
                    } else {
                        let walks: Vec<Vec<usize>> = serde_json::from_str(&value)?;
                        if walks.len() != if many { count } else { 1 } {
                            return Err("invalid walk count".into());
                        }
                        for (index, walk) in walks.iter().enumerate() {
                            if walk.len() != if many { 9 } else { 65 }
                                || walk[0] != if many { index } else { 0 }
                            {
                                return Err("invalid walk shape/order".into());
                            }
                            if walk.windows(2).any(|pair| {
                                pair[1] != (pair[0] + 1) % count && !(pair[0] == 0 && pair[1] == 2)
                            }) {
                                return Err("invalid transition".into());
                            }
                        }
                    }
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"mode":mode,"nodes":count,"edges":count+1,"samples":samples}));
        }
    }
    let path =
        std::env::var("GROGGY_PATH_PROFILE").unwrap_or_else(|_| "path_builder_steps.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release directed ring+shortcut: selected BFS/Dijkstra, staged weights, Yen node paths and seeded single/multiple walks. Fixture/spec preparation and validation excluded; five observations per workload. Includes calculation, scalar result serialization where applicable, bounded result attachment and one checked publication. Validate every distance/path/walk/publication. No matched speedup, FFI/RSS/allocation or production-scale claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote path receipt to {path}");
    Ok(())
}
