//! Shared local-RNG reservoir sampling for graph selections and pipeline masks.
use super::{GraphView, Subgraph};
use crate::{GraphError, GraphResult};

/// Algorithm R over stable IDs; seeded calls leave the shared RNG unchanged.
/// The caller supplies cancellation policy; output space is capped at population size.
pub(crate) fn sample_ids(
    ids: &[usize],
    count: usize,
    seed: Option<u64>,
    mut check: impl FnMut() -> GraphResult<()>,
) -> GraphResult<Vec<usize>> {
    check()?;
    let count = count.min(ids.len());
    let mut reservoir = Vec::new();
    reservoir.try_reserve_exact(count).map_err(|error| {
        GraphError::InvalidInput(format!("Cannot allocate node/edge sample: {error}"))
    })?;
    if count != 0 {
        let mut rng = seed.map(fastrand::Rng::with_seed).unwrap_or_default();
        for (index, &id) in ids.iter().enumerate() {
            check()?;
            if index < count {
                reservoir.push(id);
            } else {
                let replacement = rng.usize(0..=index);
                if replacement < count {
                    reservoir[replacement] = id;
                }
            }
        }
    }
    check()?;
    Ok(reservoir)
}

impl GraphView {
    /// Uniformly sample up to `count` distinct nodes and induce selected edges.
    /// Seeds reproduce results over ascending IDs; values and source pins stay frozen.
    /// Oversized counts retain the full selection; zero returns an empty selection.
    pub fn sample(&self, count: usize, seed: Option<u64>) -> GraphResult<Self> {
        if count >= self.node_count() {
            return Ok(self.clone());
        }
        if count == 0 {
            return self.select_nodes(&[]);
        }
        let nodes = sample_ids(&self.node_ids(), count, seed, || Ok(()))?;
        self.select_nodes(&nodes)
    }
}

impl Subgraph {
    /// Uniformly sample up to `count` distinct selected nodes, retaining only
    /// selected edges with both endpoints sampled. This remains a connected view;
    /// call pin_view for immutable values. Invalid live scope rejects even for zero.
    pub fn sample(&self, count: usize, seed: Option<u64>) -> GraphResult<Self> {
        self.validate_query_scope(&self.graph().borrow())?;
        if count >= self.nodes().len() {
            return Ok(self.clone());
        }
        if count == 0 {
            return Ok(Self::new(
                self.graph(),
                Default::default(),
                Default::default(),
                format!("{}_induced", self.subgraph_type()),
            ));
        }
        let nodes = sample_ids(&self.ordered_nodes(), count, seed, || Ok(()))?;
        self.induced_selection(&nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_is_checked_before_empty_sampling_and_during_reservoir_scan() {
        let mut calls = 0;
        let result = sample_ids(&(0..100).collect::<Vec<_>>(), 2, Some(7), || {
            calls += 1;
            if calls == 4 {
                Err(GraphError::InvalidInput("cancelled".into()))
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        assert_eq!(calls, 4);
        assert!(sample_ids(&[], 0, None, || {
            Err(GraphError::InvalidInput("cancelled".into()))
        })
        .is_err());
    }
}
