use std::cell::RefCell;
use std::time::{Duration, Instant};

use crate::types::{EdgeId, NodeId};

/// Compact adjacency representation in Compressed Sparse Row (CSR) form.
///
/// Neighbors for a node `i` live in `neighbors[offsets[i]..offsets[i + 1]]`.
#[derive(Debug, Default, Clone)]
pub struct Csr {
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
}

impl Csr {
    #[inline]
    pub fn node_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn neighbors(&self, idx: usize) -> &[usize] {
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        &self.neighbors[start..end]
    }
}

/// Scratch buffers reused between CSR builds to avoid repeated allocations.
#[derive(Debug, Default)]
pub struct CsrScratch {
    degree: Vec<usize>,
    cursor: Vec<usize>,
    pairs: Vec<(usize, usize)>,
}

impl CsrScratch {
    fn prepare(&mut self, node_count: usize) {
        // Always clear and resize to ensure no stale data
        self.degree.clear();
        self.degree.resize(node_count, 0);

        self.cursor.clear();
        self.cursor.resize(node_count, 0);

        self.pairs.clear();
    }
}

thread_local! {
    static SCRATCH: RefCell<CsrScratch> = RefCell::new(CsrScratch::default());
}

/// Options that control CSR construction.
#[derive(Debug, Clone, Copy)]
pub struct CsrOptions {
    /// When true, every edge (u, v) also contributes (v, u).
    pub add_reverse_edges: bool,
    /// When true, neighbors for each node are sorted in ascending order.
    pub sort_neighbors: bool,
}

impl Default for CsrOptions {
    fn default() -> Self {
        Self {
            add_reverse_edges: false,
            sort_neighbors: false,
        }
    }
}

/// Build a CSR adjacency from a subgraph edge set using thread-local scratch space.
///
/// The caller provides two passes over the edge set. Each pass receives a closure that
/// is called with `(source, target)` node identifiers already filtered to the subgraph.
pub fn build_csr_from_edges_with_scratch<I, Map, F>(
    csr: &mut Csr,
    node_count: usize,
    edges: I,
    mut index_of: Map,
    mut get_endpoints: F,
    options: CsrOptions,
) -> Duration
where
    I: Clone + IntoIterator<Item = EdgeId>,
    Map: FnMut(NodeId) -> Option<usize> + Copy,
    F: FnMut(EdgeId) -> Option<(NodeId, NodeId)>,
{
    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.prepare(node_count);

        // Pass 1: degree counting and pair caching
        let pass_start = Instant::now();
        for edge_id in edges.clone() {
            if let Some((source, target)) = get_endpoints(edge_id) {
                if let (Some(src_idx), Some(tgt_idx)) = (index_of(source), index_of(target)) {
                    scratch.pairs.push((src_idx, tgt_idx));
                    scratch.degree[src_idx] += 1;
                    if options.add_reverse_edges && src_idx != tgt_idx {
                        scratch.degree[tgt_idx] += 1;
                    }
                }
            }
        }
        let endpoint_time = pass_start.elapsed();

        // Prefix sum -> offsets
        csr.offsets.clear();
        csr.offsets.resize(node_count + 1, 0);
        let mut acc = 0usize;
        for i in 0..node_count {
            csr.offsets[i] = acc;
            acc += scratch.degree[i];
        }
        csr.offsets[node_count] = acc;

        // Prepare neighbor array
        csr.neighbors.clear();
        csr.neighbors.resize(acc, 0);

        // Reset cursors to offsets
        scratch.cursor.clone_from_slice(&csr.offsets[..node_count]);

        // Pass 2: fill neighbors from cached pairs
        let mut pairs = std::mem::take(&mut scratch.pairs);
        for &(src_idx, tgt_idx) in &pairs {
            let pos = scratch.cursor[src_idx];
            csr.neighbors[pos] = tgt_idx;
            scratch.cursor[src_idx] = pos + 1;

            if options.add_reverse_edges && src_idx != tgt_idx {
                let pos_rev = scratch.cursor[tgt_idx];
                csr.neighbors[pos_rev] = src_idx;
                scratch.cursor[tgt_idx] = pos_rev + 1;
            }
        }

        if options.sort_neighbors {
            for i in 0..node_count {
                let start = csr.offsets[i];
                let end = csr.offsets[i + 1];
                csr.neighbors[start..end].sort_unstable();
            }
        }

        pairs.clear();
        scratch.pairs = pairs;

        endpoint_time
    })
}
