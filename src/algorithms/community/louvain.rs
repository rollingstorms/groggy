use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::community::modularity::{modularity, ModularityData};
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
        if resolution <= 0.0 {
            return Err(anyhow!("resolution must be positive"));
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
                    description: "Maximum number of coarse-graining phases.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(1)),
                },
                ParameterMetadata {
                    name: "resolution".to_string(),
                    description: "Resolution parameter (currently informational).".to_string(),
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
        let _ = self.resolution; // placeholder until multi-resolution support lands
        let snapshot = CommunityGraph::from_subgraph(&subgraph)?;
        if snapshot.edges.is_empty() {
            return Ok(subgraph);
        }

        let modularity_data = ModularityData::new(&snapshot.edges);
        let adjacency = snapshot.build_adjacency();
        let mut partition: HashMap<NodeId, usize> = snapshot
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx))
            .collect();

        let epsilon = 1e-6;

        for phase in 0..self.max_phases {
            let phase_start = Instant::now();
            let mut improved = false;
            for _ in 0..self.max_iter {
                if ctx.is_cancelled() {
                    return Err(anyhow!("louvain cancelled"));
                }

                let mut changed = false;
                for &node in &snapshot.nodes {
                    let baseline = modularity(&partition, &snapshot.edges, &modularity_data);
                    let current_comm = partition[&node];

                    let mut candidate_comms: HashSet<usize> = HashSet::new();
                    candidate_comms.insert(current_comm);
                    if let Some(neighbours) = adjacency.get(&node) {
                        for &neigh in neighbours {
                            if let Some(&comm) = partition.get(&neigh) {
                                candidate_comms.insert(comm);
                            }
                        }
                    }

                    let mut best_local_comm = current_comm;
                    let mut best_local_q = baseline;
                    for &candidate in &candidate_comms {
                        if candidate == current_comm {
                            continue;
                        }
                        let mut test_partition = partition.clone();
                        test_partition.insert(node, candidate);
                        let q = modularity(&test_partition, &snapshot.edges, &modularity_data);
                        if q > best_local_q + epsilon {
                            best_local_q = q;
                            best_local_comm = candidate;
                        }
                    }

                    if best_local_comm != current_comm {
                        partition.insert(node, best_local_comm);
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }
                improved = true;
            }

            ctx.record_duration(
                format!("community.louvain.phase{}", phase),
                phase_start.elapsed(),
            );

            if !improved {
                break;
            }

            // NOTE: Full Louvain contracts communities between phases. To keep phase 2 scoped,
            // we stop after the first improvement phase. Additional phases would require
            // graph aggregation which is slated for later roadmap steps.
            break;
        }

        let mut attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            partition
                .into_iter()
                .map(|(node, community)| (node, AttrValue::Int(community as i64)))
                .collect(),
        );

        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist Louvain communities: {err}"))?;

        Ok(subgraph)
    }
}

struct CommunityGraph {
    nodes: Vec<NodeId>,
    edges: Vec<(NodeId, NodeId)>,
}

impl CommunityGraph {
    fn from_subgraph(subgraph: &Subgraph) -> Result<Self> {
        let mut nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        nodes.sort_unstable();

        let mut edge_set: HashSet<(NodeId, NodeId)> = HashSet::new();
        if nodes.len() >= 2 {
            let graph_ref = subgraph.graph();
            let graph = graph_ref.borrow();
            for &edge_id in subgraph.edge_set() {
                let (u, v) = graph.edge_endpoints(edge_id)?;
                if u == v {
                    continue;
                }
                let pair = if u <= v { (u, v) } else { (v, u) };
                edge_set.insert(pair);
            }
        }

        Ok(Self {
            nodes,
            edges: edge_set.into_iter().collect(),
        })
    }

    fn build_adjacency(&self) -> HashMap<NodeId, Vec<NodeId>> {
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &(u, v) in &self.edges {
            adjacency.entry(u).or_default().push(v);
            adjacency.entry(v).or_default().push(u);
        }
        adjacency
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Louvain::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = spec.params.get_int("max_iter").unwrap_or(20).max(1) as usize;
        let max_phases = spec.params.get_int("max_phases").unwrap_or(1).max(1) as usize;
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
