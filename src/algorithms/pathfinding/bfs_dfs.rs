use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::bfs_layers;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

fn visit_order_dfs(subgraph: &Subgraph, start: NodeId) -> Result<Vec<NodeId>> {
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
        let start = Instant::now();
        let distances = self.execute_impl(ctx, &subgraph)?;
        ctx.record_duration("pathfinding.bfs", start.elapsed());

        let mut attrs = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            distances
                .into_iter()
                .map(|(node, value)| (node, value))
                .collect::<Vec<_>>(),
        );
        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist bfs results: {err}"))?;
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
        let start = Instant::now();
        let order = self.execute_impl(ctx, &subgraph)?;
        ctx.record_duration("pathfinding.dfs", start.elapsed());

        let mut attrs = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            order.into_iter().collect::<Vec<_>>(),
        );
        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist dfs results: {err}"))?;
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
