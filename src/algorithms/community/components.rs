//! Connected Components algorithm
//!
//! Finds connected components in a graph using optimized TraversalEngine (for undirected graphs)
//! or Tarjan's algorithm (for directed graphs with strong connectivity).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::query::traversal::{TraversalEngine, TraversalOptions};
use crate::subgraphs::subgraph::{ComponentCacheComponent, ComponentCacheMode};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};

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

impl From<ComponentMode> for ComponentCacheMode {
    fn from(mode: ComponentMode) -> Self {
        match mode {
            ComponentMode::Undirected => ComponentCacheMode::Undirected,
            ComponentMode::Weak => ComponentCacheMode::Weak,
            ComponentMode::Strong => ComponentCacheMode::Strong,
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

#[derive(Debug)]
struct ComponentComputation {
    node_assignments: Arc<Vec<(NodeId, i64)>>,
    components: Arc<Vec<Vec<NodeId>>>,
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

    /// Compute undirected or weakly connected components using optimized TraversalEngine
    ///
    /// This delegates to the fast core implementation that uses BFS with zero-copy
    /// adjacency snapshot access for maximum performance.
    ///
    /// Returns Vec directly to avoid unnecessary HashMap intermediate conversion.
    fn compute_undirected_or_weak(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        nodes: &[NodeId],
    ) -> Result<ComponentComputation> {
        let cache_mode: ComponentCacheMode = self.mode.into();
        if let Some(cached) = subgraph.component_cache_get(cache_mode) {
            let components_vec: Vec<Vec<NodeId>> = cached
                .components
                .iter()
                .map(|component| component.nodes.iter().copied().collect())
                .collect();
            return Ok(ComponentComputation {
                node_assignments: cached.assignments.clone(),
                components: Arc::new(components_vec),
            });
        }

        // Use the fast TraversalEngine implementation (same as sg.connected_components())
        let mut traversal_engine = TraversalEngine::new();
        let options = TraversalOptions::default();

        let traversal_result = {
            let graph_ref = subgraph.graph();
            let graph_borrow = graph_ref.borrow();
            let pool = graph_borrow.pool();
            let space = graph_borrow.space();
            ctx.with_scoped_timer("community.connected_components.traversal", || {
                traversal_engine.connected_components_for_nodes(
                    &pool,
                    space,
                    nodes.to_vec(),
                    options,
                )
            })?
        };

        let mut node_values = Vec::with_capacity(nodes.len());
        let mut component_nodes = Vec::with_capacity(traversal_result.components.len());
        let mut cache_components = Vec::with_capacity(traversal_result.components.len());

        for (component_id, component) in traversal_result.components.into_iter().enumerate() {
            for node in &component.nodes {
                node_values.push((*node, component_id as i64));
            }

            let node_vec = component.nodes;
            let edge_vec = component.edges;

            component_nodes.push(node_vec.clone());

            cache_components.push(ComponentCacheComponent {
                nodes: Arc::new(node_vec),
                edges: Arc::new(edge_vec),
            });
        }

        let assignments_arc = Arc::new(node_values);
        let components_arc = Arc::new(component_nodes);
        if !cache_components.is_empty() {
            subgraph.component_cache_store(
                cache_mode,
                assignments_arc.clone(),
                Arc::new(cache_components),
            );
        }

        Ok(ComponentComputation {
            node_assignments: assignments_arc,
            components: components_arc,
        })
    }

    /// Compute strongly connected components using Tarjan's algorithm
    fn compute_strong(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        nodes: &[NodeId],
    ) -> Result<ComponentComputation> {
        if let Some(cached) = subgraph.component_cache_get(ComponentCacheMode::Strong) {
            let components_vec: Vec<Vec<NodeId>> = cached
                .components
                .iter()
                .map(|component| component.nodes.iter().copied().collect())
                .collect();
            return Ok(ComponentComputation {
                node_assignments: cached.assignments.clone(),
                components: Arc::new(components_vec),
            });
        }

        // Build adjacency list from edges table
        let edges_table = ctx
            .with_scoped_timer("community.connected_components.strong.edges_table", || {
                subgraph.edges_table()
            })?;
        let edge_tuples = ctx
            .with_scoped_timer("community.connected_components.strong.edge_tuples", || {
                edges_table.as_tuples()
            })?;

        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize empty adjacency lists for all nodes
        for node in nodes {
            adjacency.insert(*node, Vec::new());
        }

        // Build adjacency list from edges (only include edges where both nodes are in our subgraph)
        ctx.with_scoped_timer(
            "community.connected_components.strong.build_adjacency",
            || {
                let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
                for (_edge_id, source, target) in &edge_tuples {
                    if node_set.contains(source) && node_set.contains(target) {
                        adjacency
                            .entry(*source)
                            .or_insert_with(Vec::new)
                            .push(*target);
                    }
                }
            },
        );

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
                            neighbor, index, stack, on_stack, indices, lowlinks, components,
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
        ctx.with_scoped_timer("community.connected_components.strong.tarjan", || {
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
        });

        // Assign component IDs - return Vec directly (no HashMap!)
        let mut node_values = Vec::with_capacity(nodes.len());
        for (component_id, component) in components.iter().enumerate() {
            for &node in component {
                node_values.push((node, component_id as i64));
            }
        }
        let mut cache_components = Vec::with_capacity(components.len());
        let graph = subgraph.graph();
        let graph_ref = graph.borrow();
        let all_edges: Vec<EdgeId> = subgraph.edges().iter().copied().collect();

        for component_nodes in &components {
            let node_set: HashSet<NodeId> = component_nodes.iter().copied().collect();
            let mut edge_vec: Vec<EdgeId> = Vec::new();
            for edge_id in &all_edges {
                if let Ok((source, target)) = graph_ref.edge_endpoints(*edge_id) {
                    if node_set.contains(&source) && node_set.contains(&target) {
                        edge_vec.push(*edge_id);
                    }
                }
            }
            cache_components.push(ComponentCacheComponent {
                nodes: Arc::new(component_nodes.clone()),
                edges: Arc::new(edge_vec),
            });
        }

        drop(graph_ref);

        let assignments_arc = Arc::new(node_values);
        let components_arc = Arc::new(components);
        if !cache_components.is_empty() {
            subgraph.component_cache_store(
                ComponentCacheMode::Strong,
                assignments_arc.clone(),
                Arc::new(cache_components),
            );
        }

        Ok(ComponentComputation {
            node_assignments: assignments_arc,
            components: components_arc,
        })
    }
}

impl Algorithm for ConnectedComponents {
    fn id(&self) -> &'static str {
        "community.connected_components"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let nodes: Vec<NodeId> = ctx
            .with_scoped_timer("community.connected_components.collect_nodes", || {
                subgraph.nodes().iter().copied().collect()
            });

        // Compute components based on mode - returns assignments and component membership
        let computation = match self.mode {
            ComponentMode::Undirected | ComponentMode::Weak => {
                self.compute_undirected_or_weak(ctx, &subgraph, &nodes)?
            }
            ComponentMode::Strong => self.compute_strong(ctx, &subgraph, &nodes)?,
        };

        let ComponentComputation {
            node_assignments,
            components,
        } = computation;

        if ctx.persist_results() {
            // Convert to AttrValue in single step (no intermediate HashMap!)
            let attr_values: Vec<(NodeId, AttrValue)> = node_assignments
                .iter()
                .map(|(node, comp_id)| (*node, AttrValue::Int(*comp_id)))
                .collect();

            // Write results in bulk
            ctx.with_scoped_timer("community.connected_components.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })?;
        } else {
            ctx.add_output(
                format!("{}.components", self.id()),
                AlgorithmOutput::Components((*components).clone()),
            );
        }

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = ConnectedComponents::metadata_template();
    let id = metadata.id.clone();

    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        // Parse mode
        let mode_str = spec.params.get_text("mode").unwrap_or("undirected");
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
