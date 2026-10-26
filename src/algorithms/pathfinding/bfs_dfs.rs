use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::pathfinding::utils::checked_dijkstra_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::types::{AttrName, AttrValue, NodeId};

#[derive(Clone, Copy)]
enum TraversalKind {
    Bfs,
    Dfs,
}

impl TraversalKind {
    fn prefix(self) -> &'static str {
        match self {
            Self::Bfs => "bfs",
            Self::Dfs => "dfs",
        }
    }
    fn output_key(self) -> &'static str {
        match self {
            Self::Bfs => "pathfinding.bfs.distances",
            Self::Dfs => "pathfinding.dfs.order",
        }
    }
}

pub(crate) fn bfs_distances(
    is_cancelled: impl Fn() -> bool,
    csr: &Csr,
    nodes: &[NodeId],
    source: usize,
) -> Result<Vec<(NodeId, i64)>> {
    if is_cancelled() {
        return Err(anyhow!("bfs cancelled"));
    }
    let mut distances = vec![None; nodes.len()];
    let mut queue = VecDeque::new();
    distances[source] = Some(0usize);
    queue.push_back((source, 0usize));
    while let Some((u, depth)) = queue.pop_front() {
        if is_cancelled() {
            return Err(anyhow!("bfs cancelled"));
        }
        for &v in csr.neighbors(u) {
            if is_cancelled() {
                return Err(anyhow!("bfs cancelled"));
            }
            if distances[v].is_none() {
                let next = depth
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("BFS distance overflow"))?;
                distances[v] = Some(next);
                queue.push_back((v, next));
            }
        }
    }
    if is_cancelled() {
        return Err(anyhow!("bfs cancelled"));
    }
    let rows = nodes
        .iter()
        .copied()
        .zip(distances)
        .filter_map(|(node, distance)| distance.map(|d| (node, d)))
        .map(|(node, distance)| Ok((node, i64::try_from(distance)?)))
        .collect::<Result<_>>()?;
    if is_cancelled() {
        return Err(anyhow!("bfs cancelled"));
    }
    Ok(rows)
}

pub(crate) fn dfs_order(
    is_cancelled: impl Fn() -> bool,
    csr: &Csr,
    nodes: &[NodeId],
    source: usize,
) -> Result<Vec<(NodeId, i64)>> {
    if is_cancelled() {
        return Err(anyhow!("dfs cancelled"));
    }
    // Sort only a private copy. Shared CSR ordering may come from another caller.
    let mut sorted = csr.clone();
    for u in 0..sorted.node_count() {
        if is_cancelled() {
            return Err(anyhow!("dfs cancelled"));
        }
        sorted.neighbors[sorted.offsets[u]..sorted.offsets[u + 1]].sort_unstable();
    }
    let mut visited = vec![false; nodes.len()];
    let mut order = vec![(nodes[source], 0)];
    visited[source] = true;
    // Explicit recursive-DFS frames: at most one frame per visited node.
    let mut stack = vec![(source, 0usize)];
    while let Some((u, next)) = stack.last_mut() {
        if is_cancelled() {
            return Err(anyhow!("dfs cancelled"));
        }
        let row = sorted.neighbors(*u);
        if *next == row.len() {
            stack.pop();
            continue;
        }
        let v = row[*next];
        *next += 1;
        if !visited[v] {
            visited[v] = true;
            order.push((nodes[v], i64::try_from(order.len())?));
            stack.push((v, 0));
        }
    }
    if is_cancelled() {
        return Err(anyhow!("dfs cancelled"));
    }
    Ok(order)
}

fn execute_traversal(
    ctx: &mut Context,
    subgraph: Subgraph,
    start_attr: &AttrName,
    output_attr: &AttrName,
    kind: TraversalKind,
) -> Result<Subgraph> {
    let start = Instant::now();
    let prefix = kind.prefix();
    if ctx.is_cancelled() {
        return Err(anyhow!("{prefix} cancelled"));
    }
    checked_dijkstra_weights(ctx, &subgraph, None)?;
    let collect_start = Instant::now();
    let nodes = subgraph.ordered_nodes();
    ctx.record_call(format!("{prefix}.collect_nodes"), collect_start.elapsed());
    ctx.record_stat(format!("{prefix}.count.input_nodes"), nodes.len() as f64);
    let mut sources = Vec::new();
    for (index, &node) in nodes.iter().enumerate() {
        if ctx.node_attribute(&subgraph, node, start_attr)?.is_some() {
            sources.push(index);
        }
    }
    if sources.len() != 1 {
        return Err(anyhow!("{prefix} requires exactly one selected node carrying start attribute {start_attr:?}; found {}", sources.len()));
    }
    let source = sources[0];
    let add_reverse = subgraph.graph().borrow().is_undirected();
    let csr = if let Some(csr) = subgraph.csr_cache_get(add_reverse) {
        ctx.record_call(format!("{prefix}.csr_cache_hit"), std::time::Duration::ZERO);
        csr
    } else {
        let index_start = Instant::now();
        let index: FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        ctx.record_call(format!("{prefix}.build_indexer"), index_start.elapsed());
        let edges = subgraph.ordered_edges();
        let graph = subgraph.graph();
        let graph = graph.borrow();
        let mut csr = Csr::default();
        let build_start = Instant::now();
        let endpoints_time = build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |node| index.get(&node).copied(),
            |edge| graph.edge_endpoints(edge).ok(),
            CsrOptions {
                add_reverse_edges: add_reverse,
                sort_neighbors: false,
            },
        );
        ctx.record_call(format!("{prefix}.csr_cache_miss"), build_start.elapsed());
        ctx.record_call(format!("{prefix}.collect_edge_endpoints"), endpoints_time);
        ctx.record_call(
            format!("{prefix}.build_csr"),
            build_start.elapsed().saturating_sub(endpoints_time),
        );
        let csr = Arc::new(csr);
        subgraph.csr_cache_store(add_reverse, csr.clone());
        csr
    };
    let compute_start = Instant::now();
    let rows = match kind {
        TraversalKind::Bfs => bfs_distances(|| ctx.is_cancelled(), &csr, &nodes, source)?,
        TraversalKind::Dfs => dfs_order(|| ctx.is_cancelled(), &csr, &nodes, source)?,
    };
    ctx.record_call(format!("{prefix}.compute"), compute_start.elapsed());
    ctx.record_stat(format!("{prefix}.count.reached_nodes"), rows.len() as f64);
    ctx.emit_iteration(0, rows.len());
    if ctx.is_cancelled() {
        return Err(anyhow!("{prefix} cancelled"));
    }
    if ctx.persist_results() {
        let values = rows
            .into_iter()
            .map(|(node, value)| (node, AttrValue::Int(value)))
            .collect();
        {
            let started = Instant::now();
            let result = ctx.attach_node_column(&subgraph, output_attr.clone(), values);
            ctx.record_call(format!("{prefix}.write_attributes"), started.elapsed());
            result
        }
        .map_err(|err| anyhow!("failed to persist {prefix} results: {err}"))?;
    } else {
        ctx.add_output(kind.output_key(), AlgorithmOutput::NodeIntegers(rows));
    }
    ctx.record_duration(format!("{prefix}.total_execution"), start.elapsed());
    Ok(subgraph)
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
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description:
                        "Attribute present on exactly one selected start node; value is ignored."
                            .to_string(),
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
}

impl Algorithm for BfsTraversal {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "pathfinding.bfs"
    }
    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }
    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        execute_traversal(
            ctx,
            subgraph,
            &self.start_attr,
            &self.output_attr,
            TraversalKind::Bfs,
        )
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
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description:
                        "Attribute present on exactly one selected start node; value is ignored."
                            .to_string(),
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
}

impl Algorithm for DfsTraversal {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "pathfinding.dfs"
    }
    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }
    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        execute_traversal(
            ctx,
            subgraph,
            &self.start_attr,
            &self.output_attr,
            TraversalKind::Dfs,
        )
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

#[cfg(test)]
mod read_cancellation_tests {
    use super::*;
    use std::cell::Cell;
    #[test]
    fn bfs_and_dfs_poll_during_work_and_reject_partial_rows() {
        let csr = Csr {
            offsets: vec![0, 3, 4, 5, 5],
            neighbors: vec![1, 2, 3, 2, 3],
        };
        let nodes = [0, 1, 2, 3];
        let polls = Cell::new(0);
        let cancelled = || {
            polls.set(polls.get() + 1);
            polls.get() > 4
        };
        assert!(bfs_distances(cancelled, &csr, &nodes, 0)
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
        polls.set(0);
        let cancelled = || {
            polls.set(polls.get() + 1);
            polls.get() > 8
        };
        assert!(dfs_order(cancelled, &csr, &nodes, 0)
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
    }
}
