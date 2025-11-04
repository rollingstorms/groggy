//! Centrality-specific step primitives for efficient algorithm composition.

use crate::algorithms::{AlgorithmParamValue, Context, CostHint};
use crate::algorithms::steps::core::{Step, StepMetadata, StepScope};
use crate::state::topology::{build_csr_from_edges_with_scratch, CsrOptions};
use crate::traits::SubgraphOperations;
use crate::types::NodeId;
use anyhow::{anyhow, Context as _, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Performs a single PageRank power iteration.
/// Mirrors the native PageRank::execute kernel for performance.
/// 
/// Inputs:
///   - ranks: current rank vector
///   - out_degrees: precomputed out-degree map
///   - damping: damping factor (default 0.85)
///   - tolerance: convergence threshold (default 1e-6)
/// 
/// Outputs:
///   - updated rank vector
///   - converged: boolean scalar (1.0 if converged, 0.0 otherwise)
///   - max_diff: scalar with maximum rank change
#[derive(Clone, Debug)]
pub struct PageRankIterStep {
    pub rank_source: String,
    pub degree_source: String,
    pub rank_target: String,
    pub converged_target: String,
    pub max_diff_target: String,
    pub damping: f64,
    pub tolerance: f64,
}

impl PageRankIterStep {
    pub fn new(
        rank_source: String,
        degree_source: String,
        rank_target: String,
        converged_target: String,
        max_diff_target: String,
        damping: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            rank_source,
            degree_source,
            rank_target,
            converged_target,
            max_diff_target,
            damping,
            tolerance,
        }
    }
}

impl Step for PageRankIterStep {
    fn id(&self) -> &'static str {
        "core.pagerank_iter"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Fused PageRank power iteration (high-performance)".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let start = Instant::now();

        // === PHASE 1: Extract Input Maps ===
        let rank_map = scope
            .variables()
            .node_map(&self.rank_source)
            .context("rank_source must be a node map")?;

        let degree_map = scope
            .variables()
            .node_map(&self.degree_source)
            .context("degree_source must be a node map")?;

        // === PHASE 2: Build Dense Vectors ===
        let subgraph = scope.subgraph();
        let nodes: Vec<NodeId> = subgraph.ordered_nodes().as_ref().to_vec();
        let n = nodes.len();

        if n == 0 {
            scope
                .variables_mut()
                .set_node_map(self.rank_target.clone(), HashMap::new());
            scope
                .variables_mut()
                .set_scalar(self.converged_target.clone(), AlgorithmParamValue::Float(1.0));
            scope
                .variables_mut()
                .set_scalar(self.max_diff_target.clone(), AlgorithmParamValue::Float(0.0));
            return Ok(());
        }

        // Build node → index map
        let mut node_to_idx: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        for (idx, &nid) in nodes.iter().enumerate() {
            node_to_idx.insert(nid, idx);
        }

        // Densify ranks and degrees
        let mut rank: Vec<f64> = vec![0.0; n];
        let mut out_degree: Vec<f64> = vec![0.0; n];

        for (idx, &nid) in nodes.iter().enumerate() {
            rank[idx] = match rank_map.get(&nid) {
                Some(AlgorithmParamValue::Float(f)) => *f as f64,
                Some(AlgorithmParamValue::Int(i)) => *i as f64,
                _ => 1.0 / n as f64, // fallback
            };

            out_degree[idx] = match degree_map.get(&nid) {
                Some(AlgorithmParamValue::Float(f)) => *f as f64,
                Some(AlgorithmParamValue::Int(i)) => *i as f64,
                _ => 0.0,
            };
        }

        // === PHASE 3: Build CSR ===
        let edges: Vec<_> = subgraph.ordered_edges().as_ref().to_vec();
        let mut csr = crate::state::topology::Csr::default();

        let graph = subgraph.graph();
        let graph_ref = graph.borrow();
        let pool_ref = graph_ref.pool();
        
        let csr_duration = build_csr_from_edges_with_scratch(
            &mut csr,
            n,
            edges.into_iter(),
            |nid| node_to_idx.get(&nid).copied(),
            |eid| pool_ref.get_edge_endpoints(eid),
            CsrOptions {
                add_reverse_edges: false, // PageRank uses directional edges
                sort_neighbors: false,
            },
        );

        ctx.record_call("pr_iter.build_csr", csr_duration);

        // === PHASE 4: Power Iteration (Single Step) ===
        let iter_start = Instant::now();

        let mut next_rank: Vec<f64> = vec![0.0; n];

        // Aggregate sink mass in one pass
        let mut sink_mass = 0.0;
        for idx in 0..n {
            if out_degree[idx] == 0.0 {
                sink_mass += rank[idx];
            }
        }
        let sink_contribution = self.damping * sink_mass / n as f64;

        // Distribute rank from non-sink nodes using CSR
        for idx in 0..csr.node_count() {
            if ctx.is_cancelled() {
                return Err(anyhow!("pagerank_iter cancelled"));
            }
            if out_degree[idx] > 0.0 {
                let contrib = self.damping * rank[idx] / out_degree[idx];
                let neighbors = csr.neighbors(idx);
                for i in 0..neighbors.len() {
                    let neighbor_idx = neighbors[i];
                    next_rank[neighbor_idx] += contrib;
                }
            }
        }

        // Add teleport and sink contributions, compute residual
        let teleport_per_node = (1.0 - self.damping) / n as f64;
        let mut max_diff = 0.0;

        for idx in 0..n {
            let new_rank = teleport_per_node + next_rank[idx] + sink_contribution;
            let diff = (new_rank - rank[idx]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            next_rank[idx] = new_rank;
        }

        let converged = max_diff <= self.tolerance;

        ctx.record_call("pr_iter.compute", iter_start.elapsed());

        // === PHASE 5: Pack Results ===
        let mut result_map: HashMap<NodeId, AlgorithmParamValue> = HashMap::with_capacity(n);

        for (idx, &nid) in nodes.iter().enumerate() {
            result_map.insert(nid, AlgorithmParamValue::Float(next_rank[idx]));
        }

        scope
            .variables_mut()
            .set_node_map(self.rank_target.clone(), result_map);
        scope
            .variables_mut()
            .set_scalar(
                self.converged_target.clone(),
                AlgorithmParamValue::Float(if converged { 1.0 } else { 0.0 }),
            );
        scope
            .variables_mut()
            .set_scalar(self.max_diff_target.clone(), AlgorithmParamValue::Float(max_diff));

        ctx.record_call("pr_iter.total", start.elapsed());

        Ok(())
    }
}
