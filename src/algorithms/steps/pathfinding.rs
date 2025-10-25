//! Path utility steps for pathfinding operations.
//!
//! This module provides step primitives for common pathfinding operations:
//! - Single-source shortest paths (Dijkstra/BFS)
//! - K-shortest paths (Yen's algorithm)
//! - Random walks with optional restart

use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::{bfs_layers, dijkstra};
use crate::algorithms::{AlgorithmParamValue, Context};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

use super::core::{Step, StepMetadata, StepScope};

/// Computes single-source shortest paths from a source node to all reachable nodes.
///
/// Uses BFS for unweighted graphs or Dijkstra for weighted graphs.
/// Stores distances as a node map variable.
pub struct ShortestPathMapStep {
    source: String,                // Variable name containing source node ID (scalar)
    weight_attr: Option<AttrName>, // Optional edge weight attribute
    output: String,                // Variable name for output distances (node map)
}

impl ShortestPathMapStep {
    pub fn new(source: String, weight_attr: Option<AttrName>, output: String) -> Self {
        Self {
            source,
            weight_attr,
            output,
        }
    }

    fn resolve_source(&self, scope: &StepScope<'_>) -> Result<NodeId> {
        let value = scope.variables().scalar(&self.source)?;
        match value {
            AlgorithmParamValue::Int(id) => Ok(*id as usize),
            AlgorithmParamValue::Text(s) => s
                .parse::<usize>()
                .map_err(|_| anyhow!("cannot parse node id from text '{}'", s)),
            _ => Err(anyhow!(
                "source variable '{}' must be int or text, got {:?}",
                self.source,
                value
            )),
        }
    }
}

impl Step for ShortestPathMapStep {
    fn id(&self) -> &'static str {
        "core.shortest_path_map"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Single-source shortest paths using BFS or Dijkstra".to_string(),
            cost_hint: crate::algorithms::CostHint::Linearithmic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let source = self.resolve_source(scope)?;
        let subgraph = scope.subgraph();

        // Validate source exists in subgraph
        if !subgraph.nodes().contains(&source) {
            return Err(anyhow!("source node {} not in subgraph", source));
        }

        let distances: HashMap<NodeId, AlgorithmParamValue> =
            if let Some(weight_attr) = &self.weight_attr {
                // Weighted: use Dijkstra
                let graph_ref = subgraph.graph();
                let graph = graph_ref.borrow();
                let mut weight_map: HashMap<(NodeId, NodeId), f64> = HashMap::new();

                for &edge_id in subgraph.edge_set() {
                    if let Ok((u, v)) = graph.edge_endpoints(edge_id) {
                        if let Ok(Some(value)) = graph.get_edge_attr(edge_id, weight_attr) {
                            if let Some(weight) = match value {
                                AttrValue::Float(f) => Some(f as f64),
                                AttrValue::Int(i) => Some(i as f64),
                                _ => None,
                            } {
                                weight_map.insert((u, v), weight);
                            }
                        }
                    }
                }

                dijkstra(subgraph, source, |u, v| {
                    weight_map.get(&(u, v)).copied().unwrap_or(1.0)
                })
                .into_iter()
                .map(|(node, dist)| (node, AlgorithmParamValue::Float(dist)))
                .collect()
            } else {
                // Unweighted: use BFS
                bfs_layers(subgraph, source)
                    .into_iter()
                    .map(|(node, dist)| (node, AlgorithmParamValue::Int(dist as i64)))
                    .collect()
            };

        ctx.emit_iteration(0, distances.len());
        scope
            .variables_mut()
            .set_node_map(self.output.clone(), distances);
        Ok(())
    }
}

/// Computes k shortest paths between source and target nodes using Yen's algorithm.
///
/// Returns a list of paths, each represented as a sequence of node IDs.
pub struct KShortestPathsStep {
    source: String,                // Variable name for source node ID (scalar)
    target: String,                // Variable name for target node ID (scalar)
    k: usize,                      // Number of paths to find
    weight_attr: Option<AttrName>, // Optional edge weight attribute
    output: String,                // Variable name for output paths
}

impl KShortestPathsStep {
    pub fn new(
        source: String,
        target: String,
        k: usize,
        weight_attr: Option<AttrName>,
        output: String,
    ) -> Self {
        Self {
            source,
            target,
            k,
            weight_attr,
            output,
        }
    }

    fn resolve_node_id(&self, scope: &StepScope<'_>, var_name: &str) -> Result<NodeId> {
        let value = scope.variables().scalar(var_name)?;
        match value {
            AlgorithmParamValue::Int(id) => Ok(*id as usize),
            AlgorithmParamValue::Text(s) => s
                .parse::<usize>()
                .map_err(|_| anyhow!("cannot parse node id from text '{}'", s)),
            _ => Err(anyhow!(
                "variable '{}' must be int or text, got {:?}",
                var_name,
                value
            )),
        }
    }

    /// Dijkstra that also tracks predecessors for path reconstruction
    fn dijkstra_with_path(
        &self,
        subgraph: &Subgraph,
        source: NodeId,
        target: NodeId,
        weight_map: &HashMap<(NodeId, NodeId), f64>,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Option<(Vec<NodeId>, f64)> {
        #[derive(Copy, Clone, Debug)]
        struct State {
            cost: f64,
            node: NodeId,
        }

        impl Eq for State {}
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.node == other.node && (self.cost - other.cost).abs() <= f64::EPSILON
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.node.cmp(&other.node))
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut prev: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap: BinaryHeap<State> = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(State {
            cost: 0.0,
            node: source,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if node == target {
                // Reconstruct path
                let mut path = vec![target];
                let mut current = target;
                while let Some(&pred) = prev.get(&current) {
                    path.push(pred);
                    current = pred;
                }
                path.reverse();
                return Some((path, cost));
            }

            if cost > dist[&node] + f64::EPSILON {
                continue;
            }

            if let Ok(neighbors) = subgraph.neighbors(node) {
                for neighbor in neighbors {
                    // Skip excluded edges
                    if excluded_edges.contains(&(node, neighbor)) {
                        continue;
                    }

                    let weight = weight_map.get(&(node, neighbor)).copied().unwrap_or(1.0);
                    let next_cost = cost + weight;
                    let best = dist.get(&neighbor).copied().unwrap_or(f64::INFINITY);

                    if next_cost + f64::EPSILON < best {
                        dist.insert(neighbor, next_cost);
                        prev.insert(neighbor, node);
                        heap.push(State {
                            cost: next_cost,
                            node: neighbor,
                        });
                    }
                }
            }
        }

        None
    }

    /// Yen's algorithm for k-shortest paths
    fn yens_algorithm(
        &self,
        subgraph: &Subgraph,
        source: NodeId,
        target: NodeId,
        weight_map: &HashMap<(NodeId, NodeId), f64>,
    ) -> Vec<(Vec<NodeId>, f64)> {
        let mut result_paths: Vec<(Vec<NodeId>, f64)> = Vec::new();

        // We'll use a custom struct for the heap that wraps the path and cost
        #[derive(Debug, Clone)]
        struct PathCandidate {
            cost_millis: i64, // Cost in millis for ordering
            path: Vec<NodeId>,
            cost: f64,
        }

        impl Eq for PathCandidate {}
        impl PartialEq for PathCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.cost_millis == other.cost_millis
            }
        }
        impl Ord for PathCandidate {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Min-heap: reverse comparison
                other.cost_millis.cmp(&self.cost_millis)
            }
        }
        impl PartialOrd for PathCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut candidate_paths: BinaryHeap<PathCandidate> = BinaryHeap::new();

        // Find the first shortest path
        if let Some((path, cost)) =
            self.dijkstra_with_path(subgraph, source, target, weight_map, &HashSet::new())
        {
            result_paths.push((path, cost));
        } else {
            return result_paths; // No path exists
        }

        // Find k-1 more paths
        for k_iter in 1..self.k {
            let prev_path = &result_paths[k_iter - 1].0;

            // Iterate over spur nodes (all but last node in previous path)
            for i in 0..prev_path.len().saturating_sub(1) {
                let spur_node = prev_path[i];
                let root_path = &prev_path[0..=i];

                let mut excluded_edges = HashSet::new();

                // Exclude edges that are part of previous paths sharing the same root
                for (existing_path, _) in &result_paths {
                    if existing_path.len() > i
                        && existing_path[0..=i] == root_path[..]
                        && i + 1 < existing_path.len()
                    {
                        excluded_edges.insert((existing_path[i], existing_path[i + 1]));
                    }
                }

                // Find shortest path from spur node to target with excluded edges
                if let Some((spur_path, spur_cost)) = self.dijkstra_with_path(
                    subgraph,
                    spur_node,
                    target,
                    weight_map,
                    &excluded_edges,
                ) {
                    // Combine root path and spur path
                    let mut total_path = root_path[0..root_path.len() - 1].to_vec();
                    total_path.extend(spur_path);

                    // Calculate total cost
                    let root_cost: f64 = (0..root_path.len() - 1)
                        .map(|j| {
                            weight_map
                                .get(&(root_path[j], root_path[j + 1]))
                                .copied()
                                .unwrap_or(1.0)
                        })
                        .sum();
                    let total_cost = root_cost + spur_cost;

                    // Add to candidates
                    candidate_paths.push(PathCandidate {
                        cost_millis: (total_cost * 1000000.0) as i64,
                        path: total_path,
                        cost: total_cost,
                    });
                }
            }

            // Select the best candidate
            if let Some(candidate) = candidate_paths.pop() {
                // Avoid duplicates
                if !result_paths.iter().any(|(p, _)| p == &candidate.path) {
                    result_paths.push((candidate.path, candidate.cost));
                }
            } else {
                break; // No more paths found
            }
        }

        result_paths
    }
}

impl Step for KShortestPathsStep {
    fn id(&self) -> &'static str {
        "core.k_shortest_paths"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "K-shortest paths using Yen's algorithm".to_string(),
            cost_hint: crate::algorithms::CostHint::Quadratic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let source = self.resolve_node_id(scope, &self.source)?;
        let target = self.resolve_node_id(scope, &self.target)?;
        let subgraph = scope.subgraph();

        // Validate nodes exist
        if !subgraph.nodes().contains(&source) {
            return Err(anyhow!("source node {} not in subgraph", source));
        }
        if !subgraph.nodes().contains(&target) {
            return Err(anyhow!("target node {} not in subgraph", target));
        }

        // Build weight map
        let weight_map = if let Some(weight_attr) = &self.weight_attr {
            let graph_ref = subgraph.graph();
            let graph = graph_ref.borrow();
            let mut wm = HashMap::new();
            for &edge_id in subgraph.edge_set() {
                if let Ok((u, v)) = graph.edge_endpoints(edge_id) {
                    if let Ok(Some(value)) = graph.get_edge_attr(edge_id, weight_attr) {
                        if let Some(weight) = match value {
                            AttrValue::Float(f) => Some(f as f64),
                            AttrValue::Int(i) => Some(i as f64),
                            _ => None,
                        } {
                            wm.insert((u, v), weight);
                        }
                    }
                }
            }
            wm
        } else {
            HashMap::new() // Will default to 1.0
        };

        let paths = self.yens_algorithm(subgraph, source, target, &weight_map);

        ctx.emit_iteration(0, paths.len());

        // Convert paths to a scalar containing serialized list
        // Format: list of (path, cost) tuples
        let paths_json = serde_json::to_string(&paths).map_err(|e| anyhow!("JSON error: {}", e))?;

        scope
            .variables_mut()
            .set_scalar(self.output.clone(), AlgorithmParamValue::Text(paths_json));
        Ok(())
    }
}

/// Performs random walks starting from specified nodes.
///
/// Supports:
/// - Fixed walk length
/// - Restart probability (teleport back to start)
/// - Weighted edge transitions
pub struct RandomWalkStep {
    start_nodes: String, // Variable name for starting nodes (node map or scalar)
    length: usize,       // Walk length
    restart_prob: f64,   // Probability of restarting from source (0.0-1.0)
    weight_attr: Option<AttrName>, // Optional edge weight for biased walks
    seed: Option<u64>,   // Random seed for reproducibility
    output: String,      // Variable name for output walks
}

impl RandomWalkStep {
    pub fn new(
        start_nodes: String,
        length: usize,
        restart_prob: f64,
        weight_attr: Option<AttrName>,
        seed: Option<u64>,
        output: String,
    ) -> Self {
        Self {
            start_nodes,
            length,
            restart_prob,
            weight_attr,
            seed,
            output,
        }
    }

    fn resolve_start_nodes(&self, scope: &StepScope<'_>) -> Result<Vec<NodeId>> {
        let vars = scope.variables();

        // Try to get as node map first
        if let Ok(map) = vars.node_map(&self.start_nodes) {
            return Ok(map.keys().copied().collect());
        }

        // Otherwise try as scalar
        if let Ok(value) = vars.scalar(&self.start_nodes) {
            match value {
                AlgorithmParamValue::Int(id) => Ok(vec![*id as usize]),
                AlgorithmParamValue::Text(s) => {
                    let id: usize = s
                        .parse()
                        .map_err(|_| anyhow!("cannot parse node id from text '{}'", s))?;
                    Ok(vec![id])
                }
                _ => Err(anyhow!(
                    "start_nodes scalar variable '{}' must be int or text",
                    self.start_nodes
                )),
            }
        } else {
            Err(anyhow!(
                "start_nodes variable '{}' must be node map or scalar node ID",
                self.start_nodes
            ))
        }
    }

    fn perform_walk(
        &self,
        subgraph: &Subgraph,
        start: NodeId,
        weight_map: &HashMap<(NodeId, NodeId), f64>,
    ) -> Vec<NodeId> {
        let mut walk = vec![start];
        let mut current = start;

        for _ in 0..self.length {
            // Check restart probability
            if fastrand::f64() < self.restart_prob {
                current = start;
                walk.push(current);
                continue;
            }

            // Get neighbors
            if let Ok(neighbors) = subgraph.neighbors(current) {
                if neighbors.is_empty() {
                    break; // Dead end
                }

                // Choose next node (weighted or uniform)
                let next = if weight_map.is_empty() {
                    // Uniform random selection
                    let idx = fastrand::usize(0..neighbors.len());
                    neighbors[idx]
                } else {
                    // Weighted selection
                    let weights: Vec<f64> = neighbors
                        .iter()
                        .map(|&n| weight_map.get(&(current, n)).copied().unwrap_or(1.0))
                        .collect();
                    let total: f64 = weights.iter().sum();

                    if total <= 0.0 {
                        // Fall back to uniform
                        let idx = fastrand::usize(0..neighbors.len());
                        neighbors[idx]
                    } else {
                        let mut pick = fastrand::f64() * total;
                        let mut selected = neighbors[0];
                        for (i, &w) in weights.iter().enumerate() {
                            if pick < w {
                                selected = neighbors[i];
                                break;
                            }
                            pick -= w;
                        }
                        selected
                    }
                };

                walk.push(next);
                current = next;
            } else {
                break; // No neighbors
            }
        }

        walk
    }
}

impl Step for RandomWalkStep {
    fn id(&self) -> &'static str {
        "core.random_walk"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Random walk sequences with optional restart and weighted transitions"
                .to_string(),
            cost_hint: crate::algorithms::CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let start_nodes = self.resolve_start_nodes(scope)?;
        let subgraph = scope.subgraph();

        // Validate restart probability
        if !(0.0..=1.0).contains(&self.restart_prob) {
            return Err(anyhow!(
                "restart_prob must be in [0.0, 1.0], got {}",
                self.restart_prob
            ));
        }

        // Build weight map if needed
        let weight_map = if let Some(weight_attr) = &self.weight_attr {
            let graph_ref = subgraph.graph();
            let graph = graph_ref.borrow();
            let mut wm = HashMap::new();
            for &edge_id in subgraph.edge_set() {
                if let Ok((u, v)) = graph.edge_endpoints(edge_id) {
                    if let Ok(Some(value)) = graph.get_edge_attr(edge_id, weight_attr) {
                        if let Some(weight) = match value {
                            AttrValue::Float(f) => Some(f as f64),
                            AttrValue::Int(i) => Some(i as f64),
                            _ => None,
                        } {
                            wm.insert((u, v), weight);
                        }
                    }
                }
            }
            wm
        } else {
            HashMap::new()
        };

        // Seed the RNG if specified
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        // Perform walks
        let walks: Vec<Vec<NodeId>> = start_nodes
            .iter()
            .map(|&start| self.perform_walk(subgraph, start, &weight_map))
            .collect();

        ctx.emit_iteration(0, walks.len());

        // Serialize walks as JSON
        let walks_json = serde_json::to_string(&walks).map_err(|e| anyhow!("JSON error: {}", e))?;

        scope
            .variables_mut()
            .set_scalar(self.output.clone(), AlgorithmParamValue::Text(walks_json));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_shortest_path_map_unweighted() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let mut ctx = Context::new();
        let mut vars = super::super::core::StepVariables::default();
        vars.set_scalar("source".to_string(), AlgorithmParamValue::Int(a as i64));

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = ShortestPathMapStep::new("source".to_string(), None, "distances".to_string());
        step.apply(&mut ctx, &mut scope).unwrap();

        let distances = scope.variables().node_map("distances").unwrap();
        assert_eq!(distances.get(&a), Some(&AlgorithmParamValue::Int(0)));
        assert_eq!(distances.get(&b), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(distances.get(&c), Some(&AlgorithmParamValue::Int(2)));
    }

    #[test]
    fn test_random_walk_basic() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let mut ctx = Context::new();
        let mut vars = super::super::core::StepVariables::default();
        vars.set_scalar("source".to_string(), AlgorithmParamValue::Int(a as i64));

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = RandomWalkStep::new(
            "source".to_string(),
            10,
            0.0,
            None,
            Some(42),
            "walks".to_string(),
        );
        step.apply(&mut ctx, &mut scope).unwrap();

        let walks_json = scope.variables().scalar("walks").unwrap();
        if let AlgorithmParamValue::Text(json) = walks_json {
            let walks: Vec<Vec<NodeId>> = serde_json::from_str(json).unwrap();
            assert_eq!(walks.len(), 1);
            assert!(walks[0].len() > 1); // Should have walked
        } else {
            panic!("Expected text output");
        }
    }
}
