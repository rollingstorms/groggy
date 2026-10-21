use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::community::modularity_levels::Level;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

#[derive(Clone, Debug)]
pub struct Louvain {
    max_iter: usize,
    max_phases: usize,
    resolution: f64,
    output_attr: AttrName,
}

impl Louvain {
    pub fn new(
        max_iter: usize,
        max_phases: usize,
        resolution: f64,
        output_attr: AttrName,
    ) -> Result<Self> {
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if max_phases == 0 {
            return Err(anyhow!("max_phases must be greater than zero"));
        }
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(anyhow!("resolution must be finite and positive"));
        }
        Ok(Self {
            max_iter,
            max_phases,
            resolution,
            output_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.louvain".to_string(),
            name: "Louvain".to_string(),
            description: "Greedy modularity optimisation for community detection.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum number of node-move iterations per phase.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(20)),
                },
                ParameterMetadata {
                    name: "max_phases".to_string(),
                    description: "Maximum local-move/aggregation levels.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(10)),
                },
                ParameterMetadata {
                    name: "resolution".to_string(),
                    description: "Positive finite modularity resolution (gamma); smaller favors larger communities.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1.0)),
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store the resulting communities.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("community".to_string())),
                },
            ],
        }
    }
}

impl Algorithm for Louvain {
    fn id(&self) -> &'static str {
        "community.louvain"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start_time = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("community detection cancelled"));
        }
        if subgraph.graph().borrow().is_directed() {
            return Err(anyhow!("louvain currently requires an undirected graph"));
        }
        crate::algorithms::pathfinding::utils::checked_dijkstra_weights(&subgraph, None)?;

        let nodes = subgraph.ordered_nodes();
        let index: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        let mut edges = Vec::new();
        {
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &edge in subgraph.ordered_edges().iter() {
                let (u, v) = graph.edge_endpoints(edge)?;
                edges.push((index[&u], index[&v], 1));
            }
        }
        ctx.record_stat("louvain.count.input_nodes", nodes.len() as f64);
        ctx.record_stat("louvain.count.input_edges", edges.len() as f64);
        let mut level = Level::new(nodes.len(), edges)?;
        let mut original_to_level: Vec<_> = (0..nodes.len()).collect();
        let mut total_iterations = 0;
        let mut total_moves = 0;
        let mut phases = 0;
        for phase in 0..self.max_phases {
            if ctx.is_cancelled() {
                return Err(anyhow!("louvain cancelled"));
            }
            let phase_start = Instant::now();
            let before = level.modularity(&level.singleton_labels(), self.resolution)?;
            let partition = level.local_moves(self.resolution, self.max_iter, ctx)?;
            let after = level.modularity(&partition.labels, self.resolution)?;
            let tolerance = 64. * f64::EPSILON * (1. + before.abs().max(after.abs()));
            if after < before - tolerance {
                return Err(anyhow!("Louvain phase decreased modularity"));
            }
            for current in &mut original_to_level {
                *current = partition.labels[*current];
            }
            total_iterations += partition.iterations;
            total_moves += partition.moves;
            phases += 1;
            ctx.record_stat(format!("louvain.phase_{phase}.modularity"), after);
            ctx.record_stat(
                format!("louvain.phase_{phase}.communities"),
                partition.communities as f64,
            );
            ctx.record_call("louvain.compute.phase", phase_start.elapsed());
            let unchanged = partition.communities == partition.labels.len();
            if partition.moves == 0
                || unchanged
                || partition.communities <= 1
                || phase + 1 == self.max_phases
            {
                break;
            }
            let coarse = level.aggregate(&partition)?;
            let lifted = coarse.modularity(&coarse.singleton_labels(), self.resolution)?;
            if (lifted - after).abs() > tolerance {
                return Err(anyhow!("Louvain aggregation changed modularity"));
            }
            level = coarse;
        }
        ctx.record_stat("louvain.total_iterations", total_iterations as f64);
        ctx.record_stat("louvain.total_moves", total_moves as f64);
        ctx.record_stat("louvain.total_phases", phases as f64);
        let partition = nodes.iter().copied().zip(original_to_level).collect();
        if ctx.is_cancelled() {
            return Err(anyhow!("louvain cancelled"));
        }
        self.persist_partition(ctx, subgraph, partition, start_time)
    }
}

impl Louvain {
    fn persist_partition(
        &self,
        ctx: &mut Context,
        subgraph: Subgraph,
        partition: HashMap<NodeId, usize>,
        start_time: Instant,
    ) -> Result<Subgraph> {
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = partition
                .iter()
                .map(|(&node, &community)| (node, AttrValue::Int(community as i64)))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist Louvain communities: {err}"))?;

            ctx.record_call("louvain.write_attributes", write_start.elapsed());
        } else {
            let store_start = Instant::now();
            ctx.record_call("louvain.store_output", store_start.elapsed());
        }

        ctx.record_call("louvain.total_execution", start_time.elapsed());

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Louvain::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = usize::try_from(spec.params.get_int("max_iter").unwrap_or(20))
            .map_err(|_| anyhow!("max_iter must be positive and fit usize"))?;
        let max_phases = usize::try_from(spec.params.get_int("max_phases").unwrap_or(10))
            .map_err(|_| anyhow!("max_phases must be positive and fit usize"))?;
        let resolution = spec.params.get_float("resolution").unwrap_or(1.0);
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();
        Louvain::new(max_iter, max_phases, resolution, output_attr)
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::types::AttrName;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn louvain_separates_components() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, c).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = Louvain::new(10, 1, 1.0, "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();
        let attr_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let attr_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let attr_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        let attr_d = result.get_node_attribute(d, &attr_name).unwrap().unwrap();

        assert_eq!(attr_a, attr_b);
        assert_eq!(attr_c, attr_d);
        assert_ne!(attr_a, attr_c);
    }
}
