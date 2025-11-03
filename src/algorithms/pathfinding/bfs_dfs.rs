use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::bfs_layers;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

fn visit_order_dfs(subgraph: &Subgraph, start: NodeId) -> Result<Vec<NodeId>> {
    // Try CSR path first for optimal performance
    if let Some(csr) = subgraph.csr_cache_get(false) {
        let nodes = subgraph.ordered_nodes();
        let n = nodes.len();

        if let Some(start_idx) = nodes.iter().position(|&n| n == start) {
            let mut visited = vec![false; n];
            let mut stack = vec![start_idx];
            let mut order = Vec::new();

            while let Some(u) = stack.pop() {
                if !visited[u] {
                    visited[u] = true;
                    order.push(nodes[u]);

                    // Add neighbors in reverse order to maintain DFS semantics
                    let neighbors = csr.neighbors(u);
                    for &v in neighbors.iter().rev() {
                        if !visited[v] {
                            stack.push(v);
                        }
                    }
                }
            }

            return Ok(order);
        }
    }

    // Fallback to trait-based implementation
    let mut visited = HashSet::new();
    let mut stack = vec![start];
    let mut order = Vec::new();

    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            order.push(node);
            if let Ok(mut neighbors) = subgraph.neighbors(node) {
                neighbors.reverse();
                for neighbor in neighbors {
                    stack.push(neighbor);
                }
            }
        }
    }

    Ok(order)
}

#[derive(Clone, Debug)]
pub struct BfsTraversal {
    start_attr: AttrName,
    output_attr: AttrName,
}

impl BfsTraversal {
    pub fn new(start_attr: AttrName, output_attr: AttrName) -> Self {
        Self {
            start_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "pathfinding.bfs".to_string(),
            name: "Breadth-First Search".to_string(),
            description: "BFS traversal order and distances.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description: "Node attribute containing BFS start node (id).".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store BFS distance.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("bfs_distance".to_string())),
                },
            ],
        }
    }

    fn execute_impl(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
    ) -> Result<HashMap<NodeId, AttrValue>> {
        let start_node = subgraph
            .nodes()
            .iter()
            .find(|&&node| {
                subgraph
                    .get_node_attribute(node, &self.start_attr)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .ok_or_else(|| anyhow!("no node in subgraph carries start attribute"))?;

        let distances = bfs_layers(subgraph, *start_node);
        let mut outputs = HashMap::new();
        for (&node, &dist) in &distances {
            outputs.insert(node, AttrValue::Int(dist as i64));
        }
        ctx.emit_iteration(0, distances.len());
        Ok(outputs)
    }
}

impl Algorithm for BfsTraversal {
    fn id(&self) -> &'static str {
        "pathfinding.bfs"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("bfs.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("bfs.count.input_nodes", nodes.len() as f64);

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let mut node_to_index = rustc_hash::FxHashMap::default();
        node_to_index.reserve(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            node_to_index.insert(node, i);
        }
        ctx.record_call("bfs.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = false;
        if subgraph.csr_cache_get(add_reverse).is_some() {
            ctx.record_call("bfs.csr_cache_hit", std::time::Duration::from_nanos(0));
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
            ctx.record_call("bfs.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("bfs.build_csr", csr_time);
            subgraph.csr_cache_store(add_reverse, Arc::new(csr));
        }

        // Phase 3: Execute BFS (will use cached CSR automatically)
        let distances = self.execute_impl(ctx, &subgraph)?;

        // Phase 4: Write results
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = distances
                .iter()
                .map(|(&node, value)| (node, value.clone()))
                .collect();

            ctx.with_scoped_timer("bfs.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist bfs results: {err}"))?;
        }

        ctx.record_duration("bfs.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

#[derive(Clone, Debug)]
pub struct DfsTraversal {
    start_attr: AttrName,
    output_attr: AttrName,
}

impl DfsTraversal {
    pub fn new(start_attr: AttrName, output_attr: AttrName) -> Self {
        Self {
            start_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "pathfinding.dfs".to_string(),
            name: "Depth-First Search".to_string(),
            description: "DFS traversal ordering.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description: "Node attribute containing DFS start node (id).".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store DFS visit order.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("dfs_order".to_string())),
                },
            ],
        }
    }

    fn execute_impl(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
    ) -> Result<HashMap<NodeId, AttrValue>> {
        let start_node = subgraph
            .nodes()
            .iter()
            .find(|&&node| {
                subgraph
                    .get_node_attribute(node, &self.start_attr)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .ok_or_else(|| anyhow!("no node in subgraph carries start attribute"))?;

        let order = visit_order_dfs(subgraph, *start_node)?;
        let mut outputs = HashMap::new();
        for (idx, node) in order.iter().enumerate() {
            outputs.insert(*node, AttrValue::Int(idx as i64));
        }
        ctx.emit_iteration(0, order.len());
        Ok(outputs)
    }
}

impl Algorithm for DfsTraversal {
    fn id(&self) -> &'static str {
        "pathfinding.dfs"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("dfs.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("dfs.count.input_nodes", nodes.len() as f64);

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let mut node_to_index = rustc_hash::FxHashMap::default();
        node_to_index.reserve(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            node_to_index.insert(node, i);
        }
        ctx.record_call("dfs.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = false;
        if subgraph.csr_cache_get(add_reverse).is_some() {
            ctx.record_call("dfs.csr_cache_hit", std::time::Duration::from_nanos(0));
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
            ctx.record_call("dfs.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("dfs.build_csr", csr_time);
            subgraph.csr_cache_store(add_reverse, Arc::new(csr));
        }

        // Phase 3: Execute DFS (will use cached CSR automatically)
        let order = self.execute_impl(ctx, &subgraph)?;

        // Phase 4: Write results
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = order
                .iter()
                .map(|(&node, value)| (node, value.clone()))
                .collect();

            ctx.with_scoped_timer("dfs.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist dfs results: {err}"))?;
        }

        ctx.record_duration("dfs.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let bfs_metadata = BfsTraversal::metadata_template();
    let dfs_metadata = DfsTraversal::metadata_template();

    let bfs_id = bfs_metadata.id.clone();
    registry.register_with_metadata(bfs_id.as_str(), bfs_metadata, |spec| {
        let start_attr = spec.params.expect_text("start_attr")?.to_string();
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("bfs_distance")
            .to_string();
        Ok(
            Box::new(BfsTraversal::new(start_attr.into(), output_attr.into()))
                as Box<dyn Algorithm>,
        )
    })?;

    let dfs_id = dfs_metadata.id.clone();
    registry.register_with_metadata(dfs_id.as_str(), dfs_metadata, |spec| {
        let start_attr = spec.params.expect_text("start_attr")?.to_string();
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("dfs_order")
            .to_string();
        Ok(
            Box::new(DfsTraversal::new(start_attr.into(), output_attr.into()))
                as Box<dyn Algorithm>,
        )
    })
}
