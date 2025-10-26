//! Connected Components algorithm
//!
//! Finds connected components in a graph using Union-Find (for undirected graphs)
//! or recursive traversal (for directed graphs with strong/weak connectivity).

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use super::utils::UnionFind;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Mode for connected component detection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentMode {
    /// Undirected graph - ignores edge direction
    Undirected,
    /// Directed graph - weakly connected (ignores direction)
    Weak,
    /// Directed graph - strongly connected (respects direction)
    Strong,
}

impl ComponentMode {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "undirected" => Ok(Self::Undirected),
            "weak" => Ok(Self::Weak),
            "strong" => Ok(Self::Strong),
            _ => Err(anyhow!(
                "Invalid component mode '{}'. Use 'undirected', 'weak', or 'strong'",
                s
            )),
        }
    }

    fn to_str(&self) -> &str {
        match self {
            Self::Undirected => "undirected",
            Self::Weak => "weak",
            Self::Strong => "strong",
        }
    }
}

/// Connected Components algorithm.
///
/// Detects connected components using efficient Union-Find for undirected/weak
/// connectivity, or Tarjan's algorithm for strongly connected components.
#[derive(Clone, Debug)]
pub struct ConnectedComponents {
    mode: ComponentMode,
    output_attr: AttrName,
}

impl ConnectedComponents {
    pub fn new(mode: ComponentMode, output_attr: AttrName) -> Self {
        Self { mode, output_attr }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.connected_components".to_string(),
            name: "Connected Components".to_string(),
            description: "Find connected components using Union-Find or Tarjan's algorithm."
                .to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linear, // O(m α(n)) ≈ O(m)
            supports_cancellation: false,
            parameters: vec![
                ParameterMetadata {
                    name: "mode".to_string(),
                    description: "Component mode: 'undirected', 'weak', or 'strong'.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("undirected".to_string())),
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store component IDs.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("component".to_string())),
                },
            ],
        }
    }

    /// Compute undirected or weakly connected components using Union-Find
    fn compute_undirected_or_weak(
        &self,
        subgraph: &Subgraph,
        nodes: &[NodeId],
    ) -> Result<HashMap<NodeId, i64>> {
        // Initialize Union-Find
        let mut uf = UnionFind::new(nodes);

        // Union all edges (ignoring direction for weak connectivity)
        for node in nodes {
            if let Ok(neighbors) = subgraph.neighbors(*node) {
                for neighbor in neighbors {
                    uf.union(*node, neighbor);
                }
            }
        }

        // Get components and assign sequential IDs
        let components = uf.get_components();
        let mut node_to_component = HashMap::new();
        for (component_id, (_, members)) in components.iter().enumerate() {
            for &node in members {
                node_to_component.insert(node, component_id as i64);
            }
        }

        Ok(node_to_component)
    }

    /// Compute strongly connected components using Tarjan's algorithm
    fn compute_strong(&self, subgraph: &Subgraph, nodes: &[NodeId]) -> Result<HashMap<NodeId, i64>> {
        // Build adjacency list from edges table
        let edges_table = subgraph.edges_table()?;
        let edge_tuples = edges_table.as_tuples()?;
        
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        
        // Initialize empty adjacency lists for all nodes
        for node in nodes {
            adjacency.insert(*node, Vec::new());
        }
        
        // Build adjacency list from edges (only include edges where both nodes are in our subgraph)
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        for (_edge_id, source, target) in edge_tuples {
            if node_set.contains(&source) && node_set.contains(&target) {
                adjacency.entry(source).or_insert_with(Vec::new).push(target);
            }
        }

        // Tarjan's algorithm state
        let mut index = 0;
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlinks: HashMap<NodeId, usize> = HashMap::new();
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        // Tarjan's strongconnect function
        fn strongconnect(
            node: NodeId,
            index: &mut usize,
            stack: &mut Vec<NodeId>,
            on_stack: &mut HashSet<NodeId>,
            indices: &mut HashMap<NodeId, usize>,
            lowlinks: &mut HashMap<NodeId, usize>,
            components: &mut Vec<Vec<NodeId>>,
            adjacency: &HashMap<NodeId, Vec<NodeId>>,
        ) {
            // Set the depth index for this node
            indices.insert(node, *index);
            lowlinks.insert(node, *index);
            *index += 1;
            stack.push(node);
            on_stack.insert(node);

            // Consider successors
            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if !indices.contains_key(&neighbor) {
                        // Neighbor not yet visited; recurse
                        strongconnect(
                            neighbor,
                            index,
                            stack,
                            on_stack,
                            indices,
                            lowlinks,
                            components,
                            adjacency,
                        );
                        let neighbor_lowlink = lowlinks[&neighbor];
                        let current_lowlink = lowlinks[&node];
                        lowlinks.insert(node, current_lowlink.min(neighbor_lowlink));
                    } else if on_stack.contains(&neighbor) {
                        // Neighbor is in stack and hence in current SCC
                        let neighbor_index = indices[&neighbor];
                        let current_lowlink = lowlinks[&node];
                        lowlinks.insert(node, current_lowlink.min(neighbor_index));
                    }
                }
            }

            // If node is a root node, pop the stack and generate an SCC
            if lowlinks[&node] == indices[&node] {
                let mut component = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack.remove(&w);
                    component.push(w);
                    if w == node {
                        break;
                    }
                }
                components.push(component);
            }
        }

        // Run Tarjan's algorithm on all unvisited nodes
        for &node in nodes {
            if !indices.contains_key(&node) {
                strongconnect(
                    node,
                    &mut index,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut components,
                    &adjacency,
                );
            }
        }

        // Assign component IDs
        let mut node_to_component = HashMap::new();
        for (component_id, component) in components.iter().enumerate() {
            for &node in component {
                node_to_component.insert(node, component_id as i64);
            }
        }

        Ok(node_to_component)
    }
}

impl Algorithm for ConnectedComponents {
    fn id(&self) -> &'static str {
        "community.connected_components"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, _ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();

        // Compute components based on mode
        let node_to_component = match self.mode {
            ComponentMode::Undirected | ComponentMode::Weak => {
                self.compute_undirected_or_weak(&subgraph, &nodes)?
            }
            ComponentMode::Strong => self.compute_strong(&subgraph, &nodes)?,
        };

        // Prepare bulk attributes as HashMap<AttrName, Vec<(NodeId, AttrValue)>>
        let node_values: Vec<(NodeId, AttrValue)> = node_to_component
            .into_iter()
            .map(|(node, comp_id)| (node, AttrValue::Int(comp_id)))
            .collect();

        let mut attrs_map = HashMap::new();
        attrs_map.insert(self.output_attr.clone(), node_values);

        // Write results in bulk
        subgraph.set_node_attrs(attrs_map)?;

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = ConnectedComponents::metadata_template();
    let id = metadata.id.clone();

    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        // Parse mode
        let mode_str = spec
            .params
            .get_text("mode")
            .unwrap_or("undirected");
        let mode = ComponentMode::from_str(mode_str)?;

        // Parse output_attr
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("component")
            .to_string();

        Ok(Box::new(ConnectedComponents::new(mode, output_attr)) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::traits::GraphOperations;

    #[test]
    fn test_undirected_components() -> Result<()> {
        let mut graph = Graph::new(false)?;

        // Create two components: (1-2-3) and (4-5)
        graph.add_nodes(&[1, 2, 3, 4, 5])?;
        graph.add_edge(1, 2, None)?;
        graph.add_edge(2, 3, None)?;
        graph.add_edge(4, 5, None)?;

        let subgraph = graph.subgraph()?;
        let algo = ConnectedComponents::new(ComponentMode::Undirected, "component".to_string());
        let mut context = Context::default();

        let result = algo.execute(&mut context, subgraph)?;

        // Check that nodes in same component have same ID
        let comp1 = result.node_attr(1, "component")?.unwrap();
        let comp2 = result.node_attr(2, "component")?.unwrap();
        let comp3 = result.node_attr(3, "component")?.unwrap();
        let comp4 = result.node_attr(4, "component")?.unwrap();
        let comp5 = result.node_attr(5, "component")?.unwrap();

        assert_eq!(comp1, comp2);
        assert_eq!(comp2, comp3);
        assert_eq!(comp4, comp5);
        assert_ne!(comp1, comp4);

        Ok(())
    }

    #[test]
    fn test_strong_components() -> Result<()> {
        let mut graph = Graph::new(true)?; // directed

        // Create a cycle: 1 -> 2 -> 3 -> 1 (strongly connected)
        // And a separate node: 4
        graph.add_nodes(&[1, 2, 3, 4])?;
        graph.add_edge(1, 2, None)?;
        graph.add_edge(2, 3, None)?;
        graph.add_edge(3, 1, None)?;

        let subgraph = graph.subgraph()?;
        let algo = ConnectedComponents::new(ComponentMode::Strong, "component".to_string());
        let mut context = Context::default();

        let result = algo.execute(&mut context, subgraph)?;

        // Check that cycle nodes have same component
        let comp1 = result.node_attr(1, "component")?.unwrap();
        let comp2 = result.node_attr(2, "component")?.unwrap();
        let comp3 = result.node_attr(3, "component")?.unwrap();
        let comp4 = result.node_attr(4, "component")?.unwrap();

        assert_eq!(comp1, comp2);
        assert_eq!(comp2, comp3);
        assert_ne!(comp1, comp4); // Node 4 is separate

        Ok(())
    }

    #[test]
    fn test_weak_vs_strong() -> Result<()> {
        let mut graph = Graph::new(true)?; // directed

        // Create: 1 -> 2 -> 3 (not a cycle, only weakly connected)
        graph.add_nodes(&[1, 2, 3])?;
        graph.add_edge(1, 2, None)?;
        graph.add_edge(2, 3, None)?;

        let subgraph = graph.subgraph()?;
        let mut context = Context::default();

        // Weak connectivity: all in one component
        let weak_algo = ConnectedComponents::new(ComponentMode::Weak, "weak_comp".to_string());
        let result_weak = weak_algo.execute(&mut context, subgraph.clone())?;

        let weak1 = result_weak.node_attr(1, "weak_comp")?.unwrap();
        let weak2 = result_weak.node_attr(2, "weak_comp")?.unwrap();
        let weak3 = result_weak.node_attr(3, "weak_comp")?.unwrap();
        assert_eq!(weak1, weak2);
        assert_eq!(weak2, weak3);

        // Strong connectivity: each node is its own component
        let strong_algo =
            ConnectedComponents::new(ComponentMode::Strong, "strong_comp".to_string());
        let result_strong = strong_algo.execute(&mut context, subgraph)?;

        let strong1 = result_strong.node_attr(1, "strong_comp")?.unwrap();
        let strong2 = result_strong.node_attr(2, "strong_comp")?.unwrap();
        let strong3 = result_strong.node_attr(3, "strong_comp")?.unwrap();
        assert_ne!(strong1, strong2);
        assert_ne!(strong2, strong3);

        Ok(())
    }
}
