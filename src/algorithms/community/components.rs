//! High-Performance Connected Components Algorithm
//!
//! Implements optimized connected components detection with two modes:
//! - Weak/Undirected: BFS traversal with CSR graph for cache-friendly access
//! - Strong: Iterative Tarjan's algorithm avoiding recursion overhead
//!
//! Key optimizations:
//! - Compressed Sparse Row (CSR) format for contiguous neighbor storage
//! - Dense node indexing (u32) with fallback to sparse hash maps
//! - Vec-based BFS queue (no VecDeque)
//! - Aggressive inlining of hot-path functions
//! - Conditional output computation (only when persisting)
//! - Linear time O(V+E) with minimized constant factors
//!
//! Profiling:
//! - Detailed phase-by-phase timing with call counts
//! - Tracks node/edge processing statistics
//! - Per-component metrics for bottleneck analysis

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
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

    #[allow(dead_code)]
    #[allow(clippy::wrong_self_convention)]
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

/// High-performance node indexer: uses dense Vec when IDs are near-contiguous,
/// falls back to FxHashMap for sparse ID spaces.
/// This allows O(1) lookups for typical node ID distributions.
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

        let min = *nodes.iter().min().unwrap();
        let max = *nodes.iter().max().unwrap();
        let span = (max - min) as usize + 1;

        // Dense indexing threshold: span must be reasonable relative to node count
        if span <= nodes.len() * 3 / 2 {
            let mut indices = vec![u32::MAX; span];
            for (i, &node) in nodes.iter().enumerate() {
                indices[(node - min) as usize] = i as u32;
            }
            Self::Dense {
                min_id: min,
                indices,
            }
        } else {
            // Sparse ID space: use FxHashMap for efficient lookups
            let mut map = FxHashMap::with_capacity_and_hasher(nodes.len(), Default::default());
            for (i, &node) in nodes.iter().enumerate() {
                map.insert(node, i);
            }
            Self::Sparse(map)
        }
    }

    /// Returns the dense index for a given node ID, or None if not present.
    /// Hot path: heavily inlined for minimal overhead.
    #[inline(always)]
    fn get(&self, node: NodeId) -> Option<usize> {
        match self {
            Self::Dense { min_id, indices } => {
                let offset = node.checked_sub(*min_id)? as usize;
                if offset >= indices.len() {
                    return None;
                }
                let idx = indices[offset];
                if idx == u32::MAX {
                    None
                } else {
                    Some(idx as usize)
                }
            }
            Self::Sparse(map) => map.get(&node).copied(),
        }
    }
}

/// High-performance BFS for connected component labeling with detailed profiling.
///
/// Optimizations:
/// - Uses Vec instead of VecDeque: no ring-buffer overhead, purely sequential access
/// - Visited array eliminates redundant queue insertions
/// - Inlined neighbor iteration via CSR slicing
/// - No allocations in inner loop
///
/// Profiling:
/// - Tracks allocation time, component count, nodes/edges processed
/// - Records per-component statistics for bottleneck analysis
///
/// Returns the total number of components discovered.
fn bfs_components(csr: &Csr, labels: &mut [u32], ctx: &mut Context) -> u32 {
    let n = csr.node_count();

    // Track allocation time
    let start = Instant::now();
    let mut visited = vec![false; n];
    ctx.record_call("cc.bfs.alloc_visited", start.elapsed());

    // Reusable queue: Vec with index-based traversal (no pop_front overhead)
    let start = Instant::now();
    let mut queue = Vec::with_capacity(n.min(1024));
    ctx.record_call("cc.bfs.alloc_queue", start.elapsed());

    let mut component = 0u32;
    let mut total_nodes_processed = 0usize;
    let mut total_edges_scanned = 0usize;

    for start_node in 0..n {
        if visited[start_node] {
            continue;
        }

        // Start new component
        let comp_start = Instant::now();

        visited[start_node] = true;
        labels[start_node] = component;

        // Clear and reuse queue for this component
        queue.clear();
        queue.push(start_node);

        // Index-based traversal: no dequeue operation needed
        let mut idx = 0;
        let mut component_nodes = 0;
        let mut component_edges = 0;

        while idx < queue.len() {
            let v = queue[idx];
            idx += 1;
            component_nodes += 1;

            // Scan neighbors from CSR (contiguous memory access)
            let neighbors = csr.neighbors(v);
            component_edges += neighbors.len();

            for &nbr in neighbors {
                let w = nbr;
                if !visited[w] {
                    visited[w] = true;
                    labels[w] = component;
                    queue.push(w);
                }
            }
        }

        ctx.record_call("cc.bfs.component_traversal", comp_start.elapsed());
        total_nodes_processed += component_nodes;
        total_edges_scanned += component_edges;

        component += 1;
    }

    // Record summary statistics (stored as nanoseconds for count tracking)
    ctx.record_stat("cc.count.components", component as f64);
    ctx.record_stat("cc.count.nodes_processed", total_nodes_processed as f64);
    ctx.record_stat("cc.count.edges_scanned", total_edges_scanned as f64);

    component
}

/// Optimized iterative Tarjan's algorithm for strongly connected components with profiling.
///
/// Implementation notes:
/// - Avoids recursion to prevent stack overflow on large graphs
/// - Uses explicit call stack with neighbor iteration state
/// - Maintains index, lowlink, and on_stack arrays for linear-time traversal
/// - Each node visited exactly once, each edge examined exactly once: O(V+E)
///
/// Profiling:
/// - Tracks allocation time, recursion depth, SCC discovery
/// - Records nodes/edges processed and stack operations
///
/// Returns components as vectors of node indices (in reverse topological order).
fn tarjan_components(csr: &Csr, labels: &mut [u32], ctx: &mut Context) -> Vec<Vec<usize>> {
    let n = csr.node_count();

    // Track allocation time
    let start = Instant::now();
    let mut index_counter: u32 = 0;
    let mut stack: Vec<u32> = Vec::with_capacity(n);
    let mut on_stack = vec![false; n];
    let mut indices = vec![u32::MAX; n]; // u32::MAX = unvisited
    let mut lowlinks = vec![0u32; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    ctx.record_call("cc.tarjan.alloc_arrays", start.elapsed());

    // Call stack frame: tracks DFS state without recursion
    #[derive(Clone, Copy)]
    struct Frame {
        v: u32,
        next_neighbor: usize,
    }

    let start = Instant::now();
    let mut call_stack: Vec<Frame> = Vec::with_capacity(n);
    ctx.record_call("cc.tarjan.alloc_call_stack", start.elapsed());

    let mut total_nodes_visited = 0usize;
    let mut total_edges_examined = 0usize;
    let mut max_recursion_depth = 0usize;
    let mut stack_pushes = 0usize;
    let mut stack_pops = 0usize;

    // Outer loop: handle disconnected components
    for start_node in 0..n {
        if indices[start_node] != u32::MAX {
            continue; // Already visited
        }

        call_stack.push(Frame {
            v: start_node as u32,
            next_neighbor: 0,
        });

        // Iterative DFS with explicit stack
        while let Some(mut frame) = call_stack.pop() {
            let v = frame.v as usize;

            // Track recursion depth
            max_recursion_depth = max_recursion_depth.max(call_stack.len() + 1);

            // First visit: initialize index and lowlink
            if frame.next_neighbor == 0 && indices[v] == u32::MAX {
                indices[v] = index_counter;
                lowlinks[v] = index_counter;
                index_counter += 1;
                stack.push(v as u32);
                stack_pushes += 1;
                on_stack[v] = true;
                total_nodes_visited += 1;
            }

            let neighbors = csr.neighbors(v);
            let mut advanced = false;

            // Process neighbors starting from next_neighbor
            while frame.next_neighbor < neighbors.len() {
                let w = neighbors[frame.next_neighbor];
                total_edges_examined += 1;

                if indices[w] == u32::MAX {
                    // Unvisited neighbor: recurse (push frames)
                    frame.next_neighbor += 1;
                    call_stack.push(frame); // Resume this frame later
                    call_stack.push(Frame {
                        v: w as u32,
                        next_neighbor: 0,
                    });
                    advanced = true;
                    break;
                } else if on_stack[w] {
                    // Neighbor in current SCC path: update lowlink
                    lowlinks[v] = lowlinks[v].min(indices[w]);
                }

                frame.next_neighbor += 1;
            }

            if advanced {
                continue; // Process recursive call first
            }

            // All neighbors processed: check if v is SCC root
            let component_id = components.len() as u32;
            if lowlinks[v] == indices[v] {
                // v is root: pop SCC from stack
                let scc_start = Instant::now();
                let mut component = Vec::new();
                loop {
                    let w = stack.pop().expect("Tarjan stack underflow") as usize;
                    stack_pops += 1;
                    on_stack[w] = false;
                    labels[w] = component_id;
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                components.push(component);
                ctx.record_call("cc.tarjan.scc_extraction", scc_start.elapsed());
            }

            // Propagate lowlink to parent
            if let Some(parent) = call_stack.last_mut() {
                let parent_idx = parent.v as usize;
                lowlinks[parent_idx] = lowlinks[parent_idx].min(lowlinks[v]);
            }
        }
    }

    // Record statistics
    ctx.record_stat("cc.count.strong_components", components.len() as f64);
    ctx.record_stat("cc.count.strong_nodes_visited", total_nodes_visited as f64);
    ctx.record_stat(
        "cc.count.strong_edges_examined",
        total_edges_examined as f64,
    );
    ctx.record_stat("cc.count.max_recursion_depth", max_recursion_depth as f64);
    ctx.record_stat("cc.count.stack_pushes", stack_pushes as f64);
    ctx.record_stat("cc.count.stack_pops", stack_pops as f64);

    components
}

impl ConnectedComponents {
    pub fn new(mode: ComponentMode, output_attr: AttrName) -> Self {
        Self { mode, output_attr }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.connected_components".to_string(),
            name: "Connected Components".to_string(),
            description: "High-performance connected components using CSR graph representation \
                         with optimized BFS (weak/undirected) or iterative Tarjan's algorithm (strong)."
                .to_string(),
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Linear, // O(V+E) with optimized constants
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

    /// Compute undirected or weakly connected components using optimized CSR+BFS.
    ///
    /// Performance optimizations:
    /// - Uses cache-friendly CSR graph representation
    /// - Dense node indexing for O(1) lookups
    /// - Vec-based BFS queue (no VecDeque overhead)
    /// - Conditional edge grouping (only if persisting)
    ///
    /// Profiling:
    /// - Tracks time for each major phase: indexing, CSR build, BFS, edge assignment
    /// - Records cache hits, node counts, edge counts
    ///
    /// Returns component assignments and membership lists.
    fn compute_undirected_or_weak(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        nodes: &[NodeId],
    ) -> Result<ComponentComputation> {
        let cache_mode: ComponentCacheMode = self.mode.into();

        // Check cache
        let start = Instant::now();
        if let Some(cached) = subgraph.component_cache_get(cache_mode) {
            ctx.record_call("cc.cache_hit", start.elapsed());
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
        ctx.record_call("cc.cache_miss", start.elapsed());

        if nodes.is_empty() {
            ctx.record_call("cc.empty_input", std::time::Duration::from_nanos(0));
            return Ok(ComponentComputation {
                node_assignments: Arc::new(Vec::new()),
                components: Arc::new(Vec::new()),
            });
        }

        let persist = ctx.persist_results();
        ctx.record_stat("cc.count.input_nodes", nodes.len() as f64);

        // Build dense indexer for fast node ID to index mapping
        let start = Instant::now();
        let index_by_node = NodeIndexer::new(nodes);
        ctx.record_call("cc.build_indexer", start.elapsed());

        let edges_timer = Instant::now();
        let edges = subgraph.ordered_edges();
        ctx.record_call("cc.collect_edges", edges_timer.elapsed());
        let is_directed = {
            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            graph_ref.is_directed()
        };
        let add_reverse = !is_directed || matches!(self.mode, ComponentMode::Weak);

        let csr_arc = if let Some(cached) = subgraph.csr_cache_get(add_reverse) {
            ctx.record_call("cc.csr_cache_hit", std::time::Duration::from_nanos(0));
            cached
        } else {
            ctx.record_call("cc.csr_cache_miss", std::time::Duration::from_nanos(0));

            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool_ref = graph_ref.pool();

            let mut csr = Csr::default();
            let build_start = Instant::now();
            let endpoint_duration = build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |node_id| index_by_node.get(node_id),
                |edge_id| pool_ref.get_edge_endpoints(edge_id),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
            let total_build = build_start.elapsed();
            let core_build = total_build.saturating_sub(endpoint_duration);
            ctx.record_call("cc.build_csr", core_build);
            ctx.record_call("cc.collect_edge_endpoints", endpoint_duration);

            drop(pool_ref);
            drop(graph_ref);

            let arc = Arc::new(csr);
            subgraph.csr_cache_store(add_reverse, arc.clone());
            arc
        };

        // Record CSR stats
        ctx.record_stat("cc.count.csr_nodes", csr_arc.node_count() as f64);
        ctx.record_stat("cc.count.csr_edges", csr_arc.neighbors.len() as f64);

        // Run optimized BFS with detailed profiling
        let start = Instant::now();
        let mut labels = vec![0u32; nodes.len()];
        ctx.record_call("cc.alloc_labels", start.elapsed());

        let component_count = bfs_components(csr_arc.as_ref(), &mut labels, ctx);

        // Build component node lists
        let start = Instant::now();
        let component_count_usize = component_count as usize;
        let mut component_nodes: Vec<Vec<NodeId>> = vec![Vec::new(); component_count_usize];

        for (idx, &node) in nodes.iter().enumerate() {
            let comp_idx = labels[idx] as usize;
            if comp_idx < component_nodes.len() {
                component_nodes[comp_idx].push(node);
            }
        }
        ctx.record_call("cc.build_node_lists", start.elapsed());

        // Conditionally group edges (only if persisting results)
        let component_edge_lists = if persist {
            let start = Instant::now();
            let mut lists: Vec<Vec<EdgeId>> = vec![Vec::new(); component_nodes.len()];
            let mut edges_processed = 0usize;

            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool_ref = graph_ref.pool();

            for &edge_id in edges.iter() {
                edges_processed += 1;
                if let Some((source, target)) = pool_ref.get_edge_endpoints(edge_id) {
                    if let (Some(src_idx), Some(tgt_idx)) =
                        (index_by_node.get(source), index_by_node.get(target))
                    {
                        let comp_src = labels[src_idx] as usize;
                        let comp_tgt = labels[tgt_idx] as usize;
                        if comp_src == comp_tgt && comp_src < lists.len() {
                            lists[comp_src].push(edge_id);
                        }
                    }
                }
            }

            drop(pool_ref);
            drop(graph_ref);

            ctx.record_call("cc.assign_edges", start.elapsed());
            ctx.record_stat("cc.count.edges_processed", edges_processed as f64);

            Some(lists)
        } else {
            ctx.record_call(
                "cc.skip_edge_assignment",
                std::time::Duration::from_nanos(0),
            );
            None
        };

        // Build node assignments for output
        let start = Instant::now();
        let mut node_values = Vec::with_capacity(nodes.len());
        for (idx, &node) in nodes.iter().enumerate() {
            node_values.push((node, labels[idx] as i64));
        }
        ctx.record_call("cc.build_assignments", start.elapsed());

        let assignments_arc = Arc::new(node_values);
        let components_arc = Arc::new(component_nodes);

        // Cache results if persisting
        if persist && !components_arc.is_empty() {
            let start = Instant::now();
            let edge_lists =
                component_edge_lists.unwrap_or_else(|| vec![Vec::new(); components_arc.len()]);

            let cache_components: Vec<ComponentCacheComponent> = components_arc
                .iter()
                .zip(edge_lists.into_iter())
                .map(|(members, edges)| ComponentCacheComponent {
                    nodes: Arc::new(members.clone()),
                    edges: Arc::new(edges),
                })
                .collect();

            if !cache_components.is_empty() {
                subgraph.component_cache_store(
                    cache_mode,
                    assignments_arc.clone(),
                    Arc::new(cache_components),
                );
            }
            ctx.record_call("cc.store_cache", start.elapsed());
        }

        Ok(ComponentComputation {
            node_assignments: assignments_arc,
            components: components_arc,
        })
    }

    /// Compute strongly connected components using iterative Tarjan's algorithm.
    ///
    /// Performance optimizations:
    /// - Iterative implementation avoids recursion overhead
    /// - CSR graph for cache-friendly neighbor access
    /// - Dense node indexing for O(1) lookups
    /// - Conditional edge grouping (only if persisting)
    ///
    /// Profiling:
    /// - Tracks time for each major phase
    /// - Records cache hits, node counts, recursion depth
    ///
    /// Returns component assignments and membership lists.
    fn compute_strong(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        nodes: &[NodeId],
    ) -> Result<ComponentComputation> {
        // Check cache
        let start = Instant::now();
        if let Some(cached) = subgraph.component_cache_get(ComponentCacheMode::Strong) {
            ctx.record_call("cc.strong.cache_hit", start.elapsed());
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
        ctx.record_call("cc.strong.cache_miss", start.elapsed());

        if nodes.is_empty() {
            ctx.record_call("cc.strong.empty_input", std::time::Duration::from_nanos(0));
            return Ok(ComponentComputation {
                node_assignments: Arc::new(Vec::new()),
                components: Arc::new(Vec::new()),
            });
        }

        let persist = ctx.persist_results();
        ctx.record_stat("cc.count.strong_input_nodes", nodes.len() as f64);

        // Build dense indexer
        let start = Instant::now();
        let index_by_node = NodeIndexer::new(nodes);
        ctx.record_call("cc.strong.build_indexer", start.elapsed());

        let edges_timer = Instant::now();
        let edges = subgraph.ordered_edges();
        ctx.record_call("cc.strong.collect_edges", edges_timer.elapsed());
        let csr_arc = if let Some(cached) = subgraph.csr_cache_get(false) {
            ctx.record_call(
                "cc.strong.csr_cache_hit",
                std::time::Duration::from_nanos(0),
            );
            cached
        } else {
            ctx.record_call(
                "cc.strong.csr_cache_miss",
                std::time::Duration::from_nanos(0),
            );

            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool_ref = graph_ref.pool();

            let mut csr = Csr::default();
            let build_start = Instant::now();

            let endpoint_duration = build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |node_id| index_by_node.get(node_id),
                |edge_id| pool_ref.get_edge_endpoints(edge_id),
                CsrOptions {
                    add_reverse_edges: false,
                    sort_neighbors: false,
                },
            );
            let total_build = build_start.elapsed();
            let core_build = total_build.saturating_sub(endpoint_duration);
            ctx.record_call("cc.strong.build_csr", core_build);
            ctx.record_call("cc.strong.collect_edge_endpoints", endpoint_duration);

            drop(pool_ref);
            drop(graph_ref);

            let arc = Arc::new(csr);
            subgraph.csr_cache_store(false, arc.clone());
            arc
        };

        ctx.record_stat("cc.count.strong_csr_nodes", csr_arc.node_count() as f64);
        ctx.record_stat("cc.count.strong_csr_edges", csr_arc.neighbors.len() as f64);

        // Allocate labels
        let start = Instant::now();
        let mut labels = vec![0u32; nodes.len()];
        ctx.record_call("cc.strong.alloc_labels", start.elapsed());

        // Run optimized Tarjan's algorithm with profiling
        let components_idx = tarjan_components(csr_arc.as_ref(), &mut labels, ctx);

        // Convert index-based components to NodeId-based
        let start = Instant::now();
        let mut components: Vec<Vec<NodeId>> = Vec::with_capacity(components_idx.len());
        for indices in components_idx.iter() {
            let mut component_nodes = Vec::with_capacity(indices.len());
            for &idx in indices {
                component_nodes.push(nodes[idx]);
            }
            components.push(component_nodes);
        }
        ctx.record_call("cc.strong.convert_components", start.elapsed());

        // Build node assignments
        let start = Instant::now();
        let mut node_values = Vec::with_capacity(nodes.len());
        for (idx, &node) in nodes.iter().enumerate() {
            node_values.push((node, labels[idx] as i64));
        }
        ctx.record_call("cc.strong.build_assignments", start.elapsed());

        // Conditionally group edges (only if persisting)
        let component_edge_lists = if persist {
            let start = Instant::now();
            let mut lists: Vec<Vec<EdgeId>> = vec![Vec::new(); components.len()];
            let mut edges_processed = 0usize;

            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool_ref = graph_ref.pool();

            for &edge_id in edges.iter() {
                edges_processed += 1;
                if let Some((source, target)) = pool_ref.get_edge_endpoints(edge_id) {
                    if let (Some(src_idx), Some(tgt_idx)) =
                        (index_by_node.get(source), index_by_node.get(target))
                    {
                        let comp_src = labels[src_idx] as usize;
                        let comp_tgt = labels[tgt_idx] as usize;
                        if comp_src == comp_tgt && comp_src < lists.len() {
                            lists[comp_src].push(edge_id);
                        }
                    }
                }
            }
            ctx.record_call("cc.strong.assign_edges", start.elapsed());
            ctx.record_stat("cc.count.strong_edges_processed", edges_processed as f64);

            drop(pool_ref);
            drop(graph_ref);
            Some(lists)
        } else {
            ctx.record_call(
                "cc.strong.skip_edge_assignment",
                std::time::Duration::from_nanos(0),
            );
            None
        };
        let assignments_arc = Arc::new(node_values);
        let components_arc = Arc::new(components);

        // Cache results if persisting
        if persist && !components_arc.is_empty() {
            let start = Instant::now();
            let lists =
                component_edge_lists.unwrap_or_else(|| vec![Vec::new(); components_arc.len()]);
            let cache_components: Vec<ComponentCacheComponent> = components_arc
                .iter()
                .zip(lists.into_iter())
                .map(|(component_nodes, edges)| ComponentCacheComponent {
                    nodes: Arc::new(component_nodes.clone()),
                    edges: Arc::new(edges),
                })
                .collect();

            if !cache_components.is_empty() {
                subgraph.component_cache_store(
                    ComponentCacheMode::Strong,
                    assignments_arc.clone(),
                    Arc::new(cache_components),
                );
            }
            ctx.record_call("cc.strong.store_cache", start.elapsed());
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
        let start_overall = Instant::now();

        let nodes_arc: Arc<[NodeId]> =
            ctx.with_counted_timer("cc.collect_nodes", || subgraph.ordered_nodes());
        let nodes_slice: &[NodeId] = nodes_arc.as_ref();

        // Compute components based on mode - returns assignments and component membership
        let computation = match self.mode {
            ComponentMode::Undirected | ComponentMode::Weak => {
                self.compute_undirected_or_weak(ctx, &subgraph, nodes_slice)?
            }
            ComponentMode::Strong => self.compute_strong(ctx, &subgraph, nodes_slice)?,
        };

        let ComponentComputation {
            node_assignments,
            components,
        } = computation;

        if ctx.persist_results() {
            // Convert to AttrValue in single step (no intermediate HashMap!)
            let start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = node_assignments
                .iter()
                .map(|(node, comp_id)| (*node, AttrValue::Int(*comp_id)))
                .collect();
            ctx.record_call("cc.convert_to_attr_values", start.elapsed());

            // Write results in bulk
            let start = Instant::now();
            subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)?;
            ctx.record_call("cc.write_attributes", start.elapsed());
        } else {
            let start = Instant::now();
            ctx.add_output(
                format!("{}.components", self.id()),
                AlgorithmOutput::Components((*components).clone()),
            );
            ctx.record_call("cc.store_output", start.elapsed());
        }

        // Record total execution time
        ctx.record_call("cc.total_execution", start_overall.elapsed());

        // Print detailed profiling report if environment variable is set
        if std::env::var("GROGGY_PROFILE_CC").is_ok() {
            ctx.print_profiling_report("Connected Components");
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
    use crate::api::graph::Graph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_undirected_components() -> Result<()> {
        let graph = Rc::new(RefCell::new(Graph::new()));
        let nodes = {
            let mut g = graph.borrow_mut();
            let ids = g.add_nodes(5);
            g.add_edge(ids[0], ids[1]).unwrap();
            g.add_edge(ids[1], ids[2]).unwrap();
            g.add_edge(ids[3], ids[4]).unwrap();
            ids
        };

        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(graph.clone(), node_set, "test_cc".to_string()).unwrap();
        let algo = ConnectedComponents::new(ComponentMode::Undirected, "component".to_string());
        let mut context = Context::default();

        let result = algo.execute(&mut context, subgraph)?;

        // Check that nodes in same component have same ID
        let attr_name: AttrName = "component".to_string();
        let comp1 = result.get_node_attribute(nodes[0], &attr_name)?.unwrap();
        let comp2 = result.get_node_attribute(nodes[1], &attr_name)?.unwrap();
        let comp3 = result.get_node_attribute(nodes[2], &attr_name)?.unwrap();
        let comp4 = result.get_node_attribute(nodes[3], &attr_name)?.unwrap();
        let comp5 = result.get_node_attribute(nodes[4], &attr_name)?.unwrap();

        assert_eq!(comp1, comp2);
        assert_eq!(comp2, comp3);
        assert_eq!(comp4, comp5);
        assert_ne!(comp1, comp4);

        Ok(())
    }

    #[test]
    fn test_strong_components() -> Result<()> {
        let graph = Rc::new(RefCell::new(Graph::new_directed()));
        let nodes = {
            let mut g = graph.borrow_mut();
            let ids = g.add_nodes(4);
            g.add_edge(ids[0], ids[1]).unwrap();
            g.add_edge(ids[1], ids[2]).unwrap();
            g.add_edge(ids[2], ids[0]).unwrap();
            ids
        };

        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(graph.clone(), node_set, "test_strong".to_string()).unwrap();
        let algo = ConnectedComponents::new(ComponentMode::Strong, "component".to_string());
        let mut context = Context::default();

        let result = algo.execute(&mut context, subgraph)?;

        // Check that cycle nodes have same component
        let attr_name: AttrName = "component".to_string();
        let comp1 = result.get_node_attribute(nodes[0], &attr_name)?.unwrap();
        let comp2 = result.get_node_attribute(nodes[1], &attr_name)?.unwrap();
        let comp3 = result.get_node_attribute(nodes[2], &attr_name)?.unwrap();
        let comp4 = result.get_node_attribute(nodes[3], &attr_name)?.unwrap();

        assert_eq!(comp1, comp2);
        assert_eq!(comp2, comp3);
        assert_ne!(comp1, comp4); // Node 4 is separate

        Ok(())
    }

    #[test]
    fn test_weak_vs_strong() -> Result<()> {
        let graph = Rc::new(RefCell::new(Graph::new_directed()));
        let nodes = {
            let mut g = graph.borrow_mut();
            let ids = g.add_nodes(3);
            g.add_edge(ids[0], ids[1]).unwrap();
            g.add_edge(ids[1], ids[2]).unwrap();
            ids
        };

        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(graph.clone(), node_set, "test_weak_strong".to_string()).unwrap();
        let mut context = Context::default();

        // Weak connectivity: all in one component
        let weak_algo = ConnectedComponents::new(ComponentMode::Weak, "weak_comp".to_string());
        let result_weak = weak_algo.execute(&mut context, subgraph.clone())?;

        let weak_attr: AttrName = "weak_comp".to_string();
        let weak1 = result_weak
            .get_node_attribute(nodes[0], &weak_attr)?
            .unwrap();
        let weak2 = result_weak
            .get_node_attribute(nodes[1], &weak_attr)?
            .unwrap();
        let weak3 = result_weak
            .get_node_attribute(nodes[2], &weak_attr)?
            .unwrap();
        assert_eq!(weak1, weak2);
        assert_eq!(weak2, weak3);

        // Strong connectivity: each node is its own component
        let strong_algo =
            ConnectedComponents::new(ComponentMode::Strong, "strong_comp".to_string());
        let result_strong = strong_algo.execute(&mut context, subgraph)?;

        let strong_attr: AttrName = "strong_comp".to_string();
        let strong1 = result_strong
            .get_node_attribute(nodes[0], &strong_attr)?
            .unwrap();
        let strong2 = result_strong
            .get_node_attribute(nodes[1], &strong_attr)?
            .unwrap();
        let strong3 = result_strong
            .get_node_attribute(nodes[2], &strong_attr)?
            .unwrap();
        assert_ne!(strong1, strong2);
        assert_ne!(strong2, strong3);

        Ok(())
    }
}
