use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::collect_edge_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

const EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
pub struct AStarPathfinding {
    start_attr: AttrName,
    goal_attr: AttrName,
    weight_attr: Option<AttrName>,
    heuristic_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl AStarPathfinding {
    pub fn new(
        start_attr: AttrName,
        goal_attr: AttrName,
        weight_attr: Option<AttrName>,
        heuristic_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Self {
        Self {
            start_attr,
            goal_attr,
            weight_attr,
            heuristic_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "pathfinding.astar".to_string(),
            name: "A* Pathfinding".to_string(),
            description: "A* search with optional edge weights and heuristics.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description: "Node attribute marking the start node.".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "goal_attr".to_string(),
                    description: "Node attribute marking the goal node.".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "heuristic_attr".to_string(),
                    description: "Optional heuristic cost per node (Float/Int).".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store path order.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("path_index".to_string())),
                },
            ],
        }
    }

    fn resolve_node(&self, subgraph: &Subgraph, attr: &AttrName) -> Result<NodeId> {
        subgraph
            .nodes()
            .iter()
            .find(|&&node| {
                subgraph
                    .get_node_attribute(node, attr)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .copied()
            .ok_or_else(|| anyhow!("no node in subgraph carries attribute '{attr}'"))
    }

    fn heuristics_map(&self, subgraph: &Subgraph) -> HashMap<NodeId, f64> {
        if let Some(attr) = &self.heuristic_attr {
            let mut map = HashMap::new();
            for &node in subgraph.nodes() {
                if let Ok(Some(value)) = subgraph.get_node_attribute(node, attr) {
                    if let Some(cost) = match value {
                        AttrValue::Float(f) => Some(f as f64),
                        AttrValue::Int(i) => Some(i as f64),
                        _ => None,
                    } {
                        map.insert(node, cost);
                    }
                }
            }
            map
        } else {
            HashMap::new()
        }
    }

    fn run(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, AttrValue>> {
        let start = self.resolve_node(subgraph, &self.start_attr)?;
        let goal = self.resolve_node(subgraph, &self.goal_attr)?;
        if start == goal {
            let mut map = HashMap::new();
            map.insert(start, AttrValue::Int(0));
            return Ok(map);
        }

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));
        let heuristic_map = self.heuristics_map(subgraph);

        #[derive(Copy, Clone, Debug)]
        struct State {
            estimated: f64,
            cost: f64,
            node: NodeId,
        }
        impl Eq for State {}
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.node == other.node && (self.estimated - other.estimated).abs() <= EPS
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .estimated
                    .partial_cmp(&self.estimated)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.node.cmp(&other.node))
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut open = BinaryHeap::new();
        open.push(State {
            estimated: heuristic_map.get(&start).copied().unwrap_or(0.0),
            cost: 0.0,
            node: start,
        });

        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();
        let mut g_score: HashMap<NodeId, f64> = HashMap::new();
        g_score.insert(start, 0.0);

        while let Some(State {
            estimated: _,
            cost,
            node,
        }) = open.pop()
        {
            if ctx.is_cancelled() {
                return Err(anyhow!("astar cancelled"));
            }
            if node == goal {
                break;
            }
            if cost > g_score.get(&node).copied().unwrap_or(f64::INFINITY) + EPS {
                continue;
            }

            // Try CSR path for optimal performance
            let neighbors: Vec<NodeId> = if let Some(csr) = subgraph.csr_cache_get(false) {
                let nodes = subgraph.ordered_nodes();
                if let Some(node_idx) = nodes.iter().position(|&n| n == node) {
                    csr.neighbors(node_idx)
                        .iter()
                        .map(|&idx| nodes[idx])
                        .collect()
                } else {
                    subgraph.neighbors(node)?
                }
            } else {
                subgraph.neighbors(node)?
            };

            for neighbor in neighbors {
                let edge_weight = weight_map
                    .as_ref()
                    .and_then(|map| map.get(&(node, neighbor)).copied())
                    .unwrap_or(1.0);
                let tentative = cost + edge_weight;
                let current_best = g_score.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                if tentative + EPS < current_best {
                    came_from.insert(neighbor, node);
                    g_score.insert(neighbor, tentative);
                    let heuristic = heuristic_map.get(&neighbor).copied().unwrap_or(0.0);
                    open.push(State {
                        estimated: tentative + heuristic,
                        cost: tentative,
                        node: neighbor,
                    });
                }
            }
        }

        if !came_from.contains_key(&goal) && start != goal {
            return Err(anyhow!("goal unreachable from start"));
        }

        let mut path = Vec::new();
        let mut current = goal;
        path.push(current);
        while current != start {
            current = came_from
                .get(&current)
                .copied()
                .ok_or_else(|| anyhow!("failed to reconstruct path"))?;
            path.push(current);
        }
        path.reverse();

        let mut output = HashMap::new();
        for (idx, node) in path.iter().enumerate() {
            output.insert(*node, AttrValue::Int(idx as i64));
        }

        Ok(output)
    }
}

impl Algorithm for AStarPathfinding {
    fn id(&self) -> &'static str {
        "pathfinding.astar"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("astar.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("astar.count.input_nodes", nodes.len() as f64);

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let mut node_to_index = rustc_hash::FxHashMap::default();
        node_to_index.reserve(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            node_to_index.insert(node, i);
        }
        ctx.record_call("astar.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = false;
        if subgraph.csr_cache_get(add_reverse).is_some() {
            ctx.record_call("astar.csr_cache_hit", std::time::Duration::from_nanos(0));
        } else {
            let csr_start = Instant::now();

            let edges = subgraph.ordered_edges();
            let graph_ref = subgraph.graph();
            let graph_borrow = graph_ref.borrow();

            let mut csr = Csr::default();
            let csr_time = build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |nid| node_to_index.get(&nid).copied(),
                |eid| graph_borrow.edge_endpoints(eid).ok(),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
            ctx.record_call("astar.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("astar.build_csr", csr_time);
            subgraph.csr_cache_store(add_reverse, Arc::new(csr));
        }

        // Phase 3: Execute A* (will use cached CSR automatically)
        let path_attr = self.run(ctx, &subgraph)?;

        // Phase 4: Write results
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = path_attr
                .iter()
                .map(|(&node, value)| (node, value.clone()))
                .collect();

            ctx.with_scoped_timer("astar.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist astar path: {err}"))?;
        }

        ctx.record_duration("astar.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = AStarPathfinding::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let start_attr = spec.params.expect_text("start_attr")?.to_string();
        let goal_attr = spec.params.expect_text("goal_attr")?.to_string();
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let heuristic_attr = spec
            .params
            .get_text("heuristic_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("path_index")
            .to_string();
        Ok(Box::new(AStarPathfinding::new(
            start_attr.into(),
            goal_attr.into(),
            weight_attr,
            heuristic_attr,
            output_attr.into(),
        )) as Box<dyn Algorithm>)
    })
}
