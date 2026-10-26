use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Efficient NodeId → dense index mapper
enum NodeIndexer {
    Dense {
        min_id: NodeId,
        indices: Vec<u32>, // sentinel = u32::MAX for missing nodes
    },
    Sparse(FxHashMap<NodeId, usize>),
}

impl NodeIndexer {
    /// Constructs the optimal indexer based on node ID distribution.
    /// Uses dense array if the ID span is ≤ 1.5x node count.
    fn new(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self::Sparse(FxHashMap::default());
        }

        let min = nodes.iter().copied().min().unwrap_or(0);
        let max = nodes.iter().copied().max().unwrap_or(0);
        let span = max.checked_sub(min).and_then(|delta| delta.checked_add(1));

        // Dense indexing threshold: span must be reasonable relative to node count
        if let Some(span) = span.filter(|&span| {
            span <= nodes.len().saturating_mul(3) / 2 && nodes.len() < u32::MAX as usize
        }) {
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

/// Owned f64 scores and convergence information from a captured PageRank read.
#[derive(Clone, Debug)]
pub struct PageRankResult {
    /// Node-ID ordered scores, normalized to total mass one for nonempty inputs.
    pub scores: Vec<(NodeId, f64)>,
    /// Whether the final maximum coordinate update met the requested tolerance.
    pub converged: bool,
    /// Power iterations performed; empty input uses zero iterations.
    pub iterations: usize,
    /// Maximum coordinate update before final mass normalization (zero if empty).
    pub max_update: f64,
}

pub(crate) fn personalization_value(node: NodeId, value: Option<&AttrValue>) -> Result<f64> {
    let value = match value {
        None => 1.0,
        Some(AttrValue::Float(value)) => *value as f64,
        Some(AttrValue::Float64(value)) => *value,
        Some(AttrValue::Int(value)) => *value as f64,
        Some(AttrValue::SmallInt(value)) => *value as f64,
        Some(_) => return Err(anyhow!("node {node}: personalization must be numeric")),
    };
    if !value.is_finite() || value < 0.0 {
        return Err(anyhow!(
            "node {node}: personalization must be finite and nonnegative"
        ));
    }
    Ok(value)
}

pub(crate) fn personalization(
    nodes: &[NodeId],
    mut value: impl FnMut(NodeId) -> Result<f64>,
) -> Result<Vec<f64>> {
    let mut weights = nodes
        .iter()
        .map(|&node| value(node))
        .collect::<Result<Vec<_>>>()?;
    let total: f64 = weights.iter().sum();
    if !weights.is_empty() && (!total.is_finite() || total <= 0.0) {
        return Err(anyhow!("personalization must have a finite positive total"));
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

#[derive(Clone, Debug)]
pub struct PageRank {
    damping: f64,
    max_iter: usize,
    tolerance: f64,
    personalization_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl PageRank {
    pub fn new(
        damping: f64,
        max_iter: usize,
        tolerance: f64,
        personalization_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if !(0.0..1.0).contains(&damping) {
            return Err(anyhow!("damping must be in [0,1)"));
        }
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(anyhow!("tolerance must be finite and positive"));
        }
        Ok(Self {
            damping,
            max_iter,
            tolerance,
            personalization_attr,
            output_attr,
        })
    }

    /// Captured property used for personalization, or uniform reset if absent.
    pub fn personalization_attr(&self) -> Option<&AttrName> {
        self.personalization_attr.as_ref()
    }

    pub(crate) fn compute(
        &self,
        csr: &Csr,
        nodes: &[NodeId],
        personalization: &[f64],
        ctx: &mut Context,
        cancel: Option<&AtomicBool>,
    ) -> Result<PageRankResult> {
        if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow!("pagerank cancelled"));
        }
        let n = nodes.len();
        if n == 0 {
            return Ok(PageRankResult {
                scores: Vec::new(),
                converged: true,
                iterations: 0,
                max_update: 0.0,
            });
        }
        let damping = self.damping;
        let teleport = 1.0 - damping;
        // === PHASE 6: Precompute Out-Degrees from CSR ===
        let mut out_degree: Vec<f64> = Vec::with_capacity(csr.node_count());
        for u_idx in 0..csr.node_count() {
            out_degree.push(csr.neighbors(u_idx).len() as f64);
        }

        // === PHASE 7: Initialize Rank Vectors ===
        let init_rank = 1.0 / n as f64;
        let mut rank: Vec<f64> = vec![init_rank; n];
        let mut next_rank: Vec<f64> = vec![0.0; n];

        // === PHASE 8: Power Iteration ===
        let compute_start = Instant::now();
        let mut converged = false;
        let mut final_iteration = 0;
        let mut max_update = 0.0;

        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow!("pagerank cancelled"));
            }

            let iter_start = Instant::now();

            // Zero out next_rank buffer
            next_rank.fill(0.0);

            // Aggregate sink mass in one pass
            let mut sink_mass = 0.0;
            for idx in 0..n {
                if out_degree[idx] == 0.0 {
                    sink_mass += rank[idx];
                }
            }
            let reset_mass = teleport + damping * sink_mass;

            // Distribute rank from non-sink nodes using CSR
            for idx in 0..csr.node_count() {
                if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                    return Err(anyhow!("pagerank cancelled"));
                }
                if out_degree[idx] > 0.0 {
                    let contrib = damping * rank[idx] / out_degree[idx];
                    // CSR.neighbors returns a slice - no allocation
                    for &neighbor_idx in csr.neighbors(idx) {
                        next_rank[neighbor_idx] += contrib;
                    }
                }
            }

            // Add teleport and sink contributions, compute residual
            let mut max_diff = 0.0;
            for idx in 0..n {
                let new_rank = reset_mass * personalization[idx] + next_rank[idx];
                if !new_rank.is_finite() || new_rank < 0.0 {
                    return Err(anyhow!(
                        "nonfinite or negative PageRank score at node {}",
                        nodes[idx]
                    ));
                }
                let diff = (new_rank - rank[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                next_rank[idx] = new_rank;
            }

            // Swap buffers instead of copying
            std::mem::swap(&mut rank, &mut next_rank);

            ctx.record_call("pr.compute.iter", iter_start.elapsed());
            ctx.record_stat("pr.count.iteration", (iteration + 1) as f64);
            ctx.emit_iteration(iteration, 0);

            ctx.record_stat("pr.max_update", max_diff);
            final_iteration = iteration + 1;
            max_update = max_diff;
            if max_diff <= self.tolerance {
                converged = true;
                break;
            }
        }

        ctx.record_call("pr.compute", compute_start.elapsed());
        ctx.record_stat("pr.converged", if converged { 1.0 } else { 0.0 });
        ctx.record_stat("pr.iterations_to_converge", final_iteration as f64);

        if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow!("pagerank cancelled"));
        }
        let total: f64 = rank.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(anyhow!("invalid PageRank probability mass"));
        }
        for value in &mut rank {
            *value /= total;
        }
        ctx.record_stat("pr.mass", rank.iter().sum());

        Ok(PageRankResult {
            scores: nodes.iter().copied().zip(rank).collect(),
            converged,
            iterations: final_iteration,
            max_update,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.pagerank".to_string(),
            name: "PageRank".to_string(),
            description: "Power-iteration PageRank centrality.".to_string(),
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "damping".to_string(),
                    description: "Random reset probability (1-damping).".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(0.85)),
                },
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum number of power iterations.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(50)),
                },
                ParameterMetadata {
                    name: "tolerance".to_string(),
                    description: "Residual threshold for convergence.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1e-6)),
                },
                ParameterMetadata {
                    name: "personalization_attr".to_string(),
                    description: "Optional node attribute containing personalization weights."
                        .to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store PageRank scores.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("pagerank".to_string())),
                },
            ],
        }
    }
}

impl Algorithm for PageRank {
    fn id(&self) -> &'static str {
        "centrality.pagerank"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start_time = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("pagerank cancelled"));
        }
        crate::algorithms::pathfinding::utils::checked_dijkstra_weights(&subgraph, None)?;

        // === PHASE 1: Collect Nodes ===
        let collect_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("pr.collect_nodes", collect_start.elapsed());
        ctx.record_stat("pr.count.input_nodes", nodes.len() as f64);

        let n = nodes.len();
        if n == 0 {
            if !ctx.persist_results() {
                ctx.add_output(
                    format!("{}.scores", self.id()),
                    AlgorithmOutput::NodeScores(Vec::new()),
                );
            }
            return Ok(subgraph);
        }

        // === PHASE 2: Build Indexer ===
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("pr.build_indexer", idx_start.elapsed());

        // === PHASE 3: Collect Edges ===
        let edges_start = Instant::now();
        let edges = subgraph.ordered_edges();
        ctx.record_call("pr.collect_edges", edges_start.elapsed());
        ctx.record_stat("pr.count.input_edges", edges.len() as f64);

        // === PHASE 4: Build or Get CSR ===
        let add_reverse = subgraph.graph().borrow().is_undirected();

        let csr = match subgraph.csr_cache_get(add_reverse) {
            Some(cached) => {
                ctx.record_call("pr.csr_cache_hit", std::time::Duration::from_nanos(0));
                cached
            }
            None => {
                let cache_miss_start = Instant::now();

                let graph = subgraph.graph();
                let graph_ref = graph.borrow();
                let pool_ref = graph_ref.pool();

                let mut csr = Csr::default();
                let build_start = Instant::now();
                let endpoint_duration = build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| indexer.get(nid),
                    |eid| pool_ref.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: add_reverse,
                        sort_neighbors: false,
                    },
                );
                let total_build = build_start.elapsed();
                let core_build = total_build.saturating_sub(endpoint_duration);

                ctx.record_call("pr.csr_cache_miss", cache_miss_start.elapsed());
                ctx.record_call("pr.collect_edge_endpoints", endpoint_duration);
                ctx.record_call("pr.build_csr", core_build);

                let csr_arc = Arc::new(csr);
                subgraph.csr_cache_store(add_reverse, csr_arc.clone());
                csr_arc
            }
        };

        ctx.record_stat("pr.count.csr_nodes", csr.node_count() as f64);
        ctx.record_stat("pr.count.csr_edges", csr.neighbors.len() as f64);

        // === PHASE 5: Shared captured/live value validation and power iteration ===
        let personalization = personalization(&nodes, |node| match &self.personalization_attr {
            Some(attr) => {
                personalization_value(node, subgraph.get_node_attribute(node, attr)?.as_ref())
            }
            None => Ok(1.0),
        })?;
        let result = self.compute(&csr, &nodes, &personalization, ctx, None)?;

        // === PHASE 9: Emit Results ===
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = result
                .scores
                .into_iter()
                .map(|(node, value)| (node, AttrValue::Float64(value)))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist PageRank scores: {err}"))?;

            ctx.record_call("pr.write_attributes", write_start.elapsed());
        } else {
            let store_start = Instant::now();
            ctx.add_output(
                format!("{}.scores", self.id()),
                AlgorithmOutput::NodeScores(result.scores),
            );
            ctx.record_call("pr.store_output", store_start.elapsed());
        }

        ctx.record_call("pr.total_execution", start_time.elapsed());

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = PageRank::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let damping = spec.params.get_float("damping").unwrap_or(0.85);
        let max_iter = spec.params.get_int("max_iter").unwrap_or(50);
        if max_iter <= 0 {
            return Err(anyhow!("max_iter must be positive"));
        }
        let tolerance = spec.params.get_float("tolerance").unwrap_or(1e-6);
        let personalization_attr = spec
            .params
            .get_text("personalization_attr")
            .map(|s| s.to_string());
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("pagerank")
            .to_string();
        PageRank::new(
            damping,
            max_iter as usize,
            tolerance,
            personalization_attr.map(AttrName::from),
            output_attr.into(),
        )
        .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
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
    fn pagerank_handles_small_graph() {
        let mut graph = Graph::new_manual();
        let a = graph.add_node().unwrap();
        let b = graph.add_node().unwrap();
        let c = graph.add_node().unwrap();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = PageRank::new(0.85, 30, 1e-6, None, "pagerank".into()).unwrap();
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr = result
            .get_node_attribute(a, &"pagerank".into())
            .unwrap()
            .unwrap();
        if let AttrValue::Float64(score) = attr {
            assert!(score > 0.0);
        } else {
            panic!("expected float score");
        }
    }
}
