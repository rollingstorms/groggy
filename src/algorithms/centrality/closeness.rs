use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::pathfinding::utils::{bfs_layers_csr, collect_edge_weights, dijkstra};
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Efficient NodeId → dense index mapper
enum NodeIndexer {
    Dense { min_id: NodeId, indices: Vec<u32> },
    Sparse(FxHashMap<NodeId, usize>),
}

impl NodeIndexer {
    fn new(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self::Sparse(FxHashMap::default());
        }

        let min = *nodes.iter().min().unwrap();
        let max = *nodes.iter().max().unwrap();
        let span = (max - min) as usize + 1;

        if span <= nodes.len() * 3 / 2 {
            let mut indices = vec![u32::MAX; span];
            for (i, &node) in nodes.iter().enumerate() {
                indices[(node - min) as usize] = i as u32;
            }
            Self::Dense {
                min_id: min,
                indices,
            }
        } else {
            let mut map = FxHashMap::default();
            map.reserve(nodes.len());
            for (i, &node) in nodes.iter().enumerate() {
                map.insert(node, i);
            }
            Self::Sparse(map)
        }
    }

    fn get(&self, node: NodeId) -> Option<usize> {
        match self {
            Self::Dense { min_id, indices } => {
                let offset = node.checked_sub(*min_id)? as usize;
                indices.get(offset).and_then(|&idx| {
                    if idx == u32::MAX {
                        None
                    } else {
                        Some(idx as usize)
                    }
                })
            }
            Self::Sparse(map) => map.get(&node).copied(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClosenessCentrality {
    harmonic: bool,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl ClosenessCentrality {
    pub fn new(harmonic: bool, weight_attr: Option<AttrName>, output_attr: AttrName) -> Self {
        Self {
            harmonic,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.closeness".to_string(),
            name: "Closeness Centrality".to_string(),
            description: "Closeness (classic or harmonic) centrality via BFS/Dijkstra.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "harmonic".to_string(),
                    description: "Use harmonic closeness (sum of reciprocal distances)."
                        .to_string(),
                    value_type: ParameterType::Bool,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Bool(true)),
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store closeness scores.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("closeness".to_string())),
                },
            ],
        }
    }

    fn compute(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        csr: &Csr,
        nodes: &[NodeId],
    ) -> Result<HashMap<NodeId, f64>> {
        let n = nodes.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        let mut scores: Vec<f64> = vec![0.0; n];

        // Pre-allocate buffers for BFS (reused across all sources!)
        let mut distance_buf = vec![usize::MAX; n];
        let mut queue_buf = VecDeque::with_capacity(n);

        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("closeness cancelled"));
            }

            let mut sum = 0.0;
            let mut reachable = 0;

            if let Some(ref weights) = weight_map {
                // Weighted case: use smart dijkstra() utility (auto-detects CSR)
                let dist = dijkstra(subgraph, source, |u, v| {
                    weights.get(&(u, v)).copied().unwrap_or(1.0)
                });

                for (&target, &d) in &dist {
                    if target == source {
                        continue;
                    }
                    if self.harmonic {
                        if d > 0.0 {
                            sum += 1.0 / d;
                        }
                    } else {
                        sum += d;
                        reachable += 1;
                    }
                }
            } else {
                // Unweighted case: use CSR-optimized BFS
                bfs_layers_csr(csr, nodes, idx, &mut distance_buf, &mut queue_buf);

                // Sum distances from distance array
                for i in 0..n {
                    if i == idx {
                        continue;
                    }
                    let d = distance_buf[i];
                    if d < usize::MAX {
                        if self.harmonic {
                            sum += 1.0 / (d as f64);
                        } else {
                            sum += d as f64;
                            reachable += 1;
                        }
                    }
                }
            }

            let score = if self.harmonic {
                sum
            } else if reachable > 0 {
                (reachable as f64) / sum
            } else {
                0.0
            };
            scores[idx] = score;
            ctx.emit_iteration(idx, 0);
        }

        // Convert Vec → HashMap for result
        let result: HashMap<NodeId, f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, scores[i]))
            .collect();

        Ok(result)
    }
}

impl Algorithm for ClosenessCentrality {
    fn id(&self) -> &'static str {
        "centrality.closeness"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("closeness.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("closeness.count.input_nodes", nodes.len() as f64);

        let n = nodes.len();
        if n == 0 {
            return Ok(subgraph);
        }

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("closeness.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = false;
        let csr = if let Some(cached) = subgraph.csr_cache_get(add_reverse) {
            ctx.record_call(
                "closeness.csr_cache_hit",
                std::time::Duration::from_nanos(0),
            );
            cached
        } else {
            let csr_start = Instant::now();

            let edges = subgraph.ordered_edges();
            let graph_ref = subgraph.graph();
            let graph_borrow = graph_ref.borrow();

            let mut csr_new = Csr::default();
            let csr_time = build_csr_from_edges_with_scratch(
                &mut csr_new,
                nodes.len(),
                edges.iter().copied(),
                |nid| indexer.get(nid),
                |eid| graph_borrow.edge_endpoints(eid).ok(),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
            ctx.record_call("closeness.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("closeness.build_csr", csr_time);

            let csr_arc = Arc::new(csr_new);
            subgraph.csr_cache_store(add_reverse, csr_arc.clone());
            csr_arc
        };

        ctx.record_stat("closeness.count.csr_edges", csr.neighbors.len() as f64);

        // Phase 4: Compute closeness (uses CSR-optimized BFS/Dijkstra)
        let compute_start = Instant::now();
        let scores = self.compute(ctx, &subgraph, &csr, &nodes)?;
        ctx.record_call("closeness.compute", compute_start.elapsed());

        // Phase 5: Write results
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = scores
                .iter()
                .map(|(&node, score)| (node, AttrValue::Float(*score as f32)))
                .collect();

            ctx.with_scoped_timer("closeness.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist closeness scores: {err}"))?;
        }

        ctx.record_duration("closeness.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = ClosenessCentrality::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let harmonic = spec.params.get_bool("harmonic").unwrap_or(true);
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("closeness")
            .to_string();
        Ok(Box::new(ClosenessCentrality::new(
            harmonic,
            weight_attr,
            output_attr.into(),
        )) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::traits::SubgraphOperations;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn closeness_prefers_central_node() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, b).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = ClosenessCentrality::new(true, None, "closeness".into());
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "closeness".to_string();
        let score_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let score_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let score_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();

        match (score_b, score_a, score_c) {
            (AttrValue::Float(sb), AttrValue::Float(sa), AttrValue::Float(sc)) => {
                assert!(sb > sa);
                assert!(sb > sc);
            }
            _ => panic!("expected float results"),
        }
    }
}
