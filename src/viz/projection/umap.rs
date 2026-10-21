//! Checked native neighbor layout; inspired by UMAP, not its fuzzy-set optimizer.
use super::neighborhood::{coordinates, invalid, positions, scaled_features, squared_distance};
use super::ProjectionEngine;
use crate::viz::streaming::data_source::Position;
use crate::{Graph, GraphMatrix, GraphResult};

type Spring = (usize, usize, f64);
type Pair = (usize, usize);

/// UMAP-inspired weighted-neighbor layout with sampled smooth repulsion.
/// This is not an implementation of UMAP fuzzy union or cross-entropy fitting.
#[derive(Debug)]
pub struct UMAPProjection {
    n_neighbors: usize,
    min_dist: f64,
    n_epochs: usize,
    negative_sample_rate: f64,
    seed: Option<u64>,
}
impl UMAPProjection {
    /// Configure the native inspired layout, retaining the historical constructor.
    pub fn new(
        n_neighbors: usize,
        min_dist: f64,
        n_epochs: usize,
        negative_sample_rate: f64,
    ) -> Self {
        Self {
            n_neighbors,
            min_dist,
            n_epochs,
            negative_sample_rate,
            seed: None,
        }
    }
    /// Seed local initialization and pair sampling without changing global RNG state.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Return owned coordinates in feature-row order, retaining optional row labels.
    /// The position-vector trait's graph argument does not contribute topology.
    pub fn project_matrix(&self, input: &GraphMatrix) -> GraphResult<GraphMatrix> {
        if self.n_neighbors == 0
            || !self.min_dist.is_finite()
            || self.min_dist < 0.0
            || !self.negative_sample_rate.is_finite()
            || self.negative_sample_rate < 0.0
        {
            return Err(invalid("UMAP-inspired layout requires positive neighbors, finite nonnegative min_dist and negative_sample_rate"));
        }
        let features = scaled_features(input)?;
        let n = input.shape().0;
        let requested = (n as f64 * self.negative_sample_rate).floor();
        if !requested.is_finite() || requested >= usize::MAX as f64 {
            return Err(invalid("UMAP-inspired negative-pair count overflow"));
        }
        let count = requested as usize;
        if count
            .checked_mul(std::mem::size_of::<Pair>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(invalid(
                "UMAP-inspired negative-pair allocation size overflow",
            ));
        }
        let springs = neighbors(&features, input.shape().1, n, self.n_neighbors)?;
        let mut rng = self
            .seed
            .map(fastrand::Rng::with_seed)
            .unwrap_or_else(fastrand::Rng::new);
        if n == 1 {
            return coordinates(input, vec![0.0, 0.0], "umap");
        }
        let mut data: Vec<_> = (0..n * 2).map(|_| rng.f64() * 2.0 - 1.0).collect();
        for epoch in 0..self.n_epochs {
            let mut negatives = Vec::new();
            negatives
                .try_reserve_exact(count)
                .map_err(|_| invalid("Cannot allocate UMAP-inspired negative pairs"))?;
            for _ in 0..count {
                let i = rng.usize(0..n);
                let other = rng.usize(0..n - 1);
                let j = if other >= i { other + 1 } else { other };
                negatives.push((i, j));
            }
            // Hold sampled pairs fixed during each checked optimization step.
            let (energy, gradient) = objective(&data, &springs, &negatives, self.min_dist)?;
            if gradient.iter().all(|&g| g == 0.0) {
                continue;
            }
            let mut step = 0.01 * (1.0 - epoch as f64 / self.n_epochs as f64);
            let mut accepted = None;
            for _ in 0..64 {
                let next: Vec<_> = data
                    .iter()
                    .zip(&gradient)
                    .map(|(x, g)| x - step * g)
                    .collect();
                if next == data {
                    accepted = Some(next);
                    break;
                }
                if let Ok((candidate, _)) = objective(&next, &springs, &negatives, self.min_dist) {
                    if candidate <= energy {
                        accepted = Some(next);
                        break;
                    }
                }
                step *= 0.5;
            }
            data = accepted.ok_or_else(|| {
                invalid("UMAP-inspired line search failed to find a finite nonincreasing step")
            })?;
        }
        coordinates(input, data, "umap")
    }
}
impl ProjectionEngine for UMAPProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        positions(self.project_matrix(embedding)?)
    }
    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_interpolation(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "umap"
    }
}

/// Exact directed kNN with row-index tie breaks and locally scaled similarity weights.
fn neighbors(features: &[f64], cols: usize, n: usize, k: usize) -> GraphResult<Vec<Spring>> {
    let mut springs = Vec::new();
    for i in 0..n {
        let mut ordered: Vec<_> = (0..n)
            .filter(|&j| j != i)
            .map(|j| Ok((j, squared_distance(features, cols, i, j)?.sqrt())))
            .collect::<GraphResult<Vec<_>>>()?;
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        ordered.truncate(k.min(n - 1));
        let Some(&(_, nearest)) = ordered.first() else {
            continue;
        };
        let farthest = ordered.iter().map(|&(_, d)| d).fold(nearest, f64::max);
        for (j, distance) in ordered {
            let weight = if farthest > nearest {
                (-(distance - nearest) / (farthest - nearest)).exp()
            } else {
                1.0
            };
            springs.push((i, j, weight));
        }
    }
    Ok(springs)
}

/// L = sum(0.5*w*max(r-min_dist,0)^2) + sum(0.001/sqrt(r²+1e-6)).
/// Every spring/sample updates both endpoints; sampled multiplicity is retained.
fn objective(
    data: &[f64],
    springs: &[Spring],
    negatives: &[Pair],
    min_dist: f64,
) -> GraphResult<(f64, Vec<f64>)> {
    if data.iter().any(|x| !x.is_finite()) {
        return Err(invalid("UMAP-inspired coordinates must be finite"));
    }
    let mut energy = 0.0;
    let mut gradient = vec![0.0; data.len()];
    let mut apply = |i: usize, j: usize, weight: f64, negative: bool| -> GraphResult<()> {
        let dx = data[2 * i] - data[2 * j];
        let dy = data[2 * i + 1] - data[2 * j + 1];
        let squared = dx * dx + dy * dy;
        if !squared.is_finite() {
            return Err(invalid("UMAP-inspired coordinate distance overflow"));
        }
        let coefficient = if negative {
            let soft = squared + 1e-6;
            let term = 0.001 / soft.sqrt();
            energy += term;
            -term / soft
        } else {
            let distance = squared.sqrt();
            if distance <= min_dist {
                return Ok(());
            }
            let gap = distance - min_dist;
            energy += 0.5 * weight * gap * gap;
            weight * (gap / distance)
        };
        for (axis, delta) in [(0, dx), (1, dy)] {
            let force = coefficient * delta;
            gradient[2 * i + axis] += force;
            gradient[2 * j + axis] -= force;
        }
        Ok(())
    };
    for &(i, j, w) in springs {
        apply(i, j, w, false)?;
    }
    for &(i, j) in negatives {
        apply(i, j, 0.0, true)?;
    }
    if !energy.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
        return Err(invalid("UMAP-inspired objective or gradient overflow"));
    }
    Ok((energy, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_neighbors_have_stable_ties_local_weights_and_no_self_edges() {
        // One-dimensional coordinates 0,1,3,3; ties use original row order.
        let features = [0., 1., 3., 3.];
        let springs = neighbors(&features, 1, 4, 2).unwrap();
        assert_eq!(springs.len(), 8);
        assert_eq!(springs[0], (0, 1, 1.0));
        assert_eq!(springs[1], (0, 2, (-1.0_f64).exp()));
        assert_eq!(springs[4], (2, 3, 1.0));
        assert_eq!(springs[5], (2, 1, (-1.0_f64).exp()));
        assert!(springs
            .iter()
            .all(|&(i, j, w)| i != j && w > 0.0 && w <= 1.0));
        assert_eq!(neighbors(&features, 1, 4, usize::MAX).unwrap().len(), 12);
        assert!(neighbors(&[0.0], 1, 1, 3).unwrap().is_empty());
    }
    #[test]
    fn weighted_hinge_and_repulsion_gradients_match_independent_energy() {
        let springs = [(0, 1, 0.4), (1, 0, 0.8), (2, 3, 1.0)];
        let negatives = [(0, 2), (0, 2), (1, 3)];
        let data = [0.1, 0.3, 1.2, -0.4, -0.6, 0.8, 0.7, 0.4];
        for min in [0.0, 0.5, 3.0] {
            let reference = |x: &[f64]| {
                let radius =
                    |i: usize, j: usize| (x[2 * i] - x[2 * j]).hypot(x[2 * i + 1] - x[2 * j + 1]);
                springs
                    .iter()
                    .map(|&(i, j, w)| 0.5 * w * (radius(i, j) - min).max(0.0).powi(2))
                    .sum::<f64>()
                    + negatives
                        .iter()
                        .map(|&(i, j)| 0.001 / (radius(i, j).powi(2) + 1e-6).sqrt())
                        .sum::<f64>()
            };
            let (energy, gradient) = objective(&data, &springs, &negatives, min).unwrap();
            assert!((energy - reference(&data)).abs() < 1e-14);
            for axis in 0..data.len() {
                let mut plus = data;
                let mut minus = data;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                assert!(
                    (gradient[axis] - (reference(&plus) - reference(&minus)) / 2e-6).abs() < 1e-8
                );
            }
            for axis in 0..2 {
                assert!((0..4).map(|i| gradient[2 * i + axis]).sum::<f64>().abs() < 1e-14);
            }
        }
        let coincident = objective(&[0.0; 4], &[(0, 1, 1.0)], &[(0, 1)], 0.0).unwrap();
        assert_eq!(coincident.1, [0.0; 4]);
        assert!(objective(&[f64::MAX, 0.0, -f64::MAX, 0.0], &[(0, 1, 1.0)], &[], 0.0).is_err());
    }
}
