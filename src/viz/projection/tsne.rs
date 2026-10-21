//! Dense exact t-SNE affinities and a bounded native gradient optimizer.
use super::neighborhood::{coordinates, invalid, positions, squared_distances};
use super::ProjectionEngine;
use crate::viz::streaming::data_source::Position;
use crate::{Graph, GraphMatrix, GraphResult};

/// Dense t-SNE projection with entropy-calibrated symmetric affinities.
#[derive(Debug)]
pub struct TSNEProjection {
    perplexity: f64,
    iterations: usize,
    learning_rate: f64,
    early_exaggeration: f64,
    seed: Option<u64>,
}
impl TSNEProjection {
    /// Configure exact dense affinities and bounded gradient descent.
    pub fn new(
        perplexity: f64,
        iterations: usize,
        learning_rate: f64,
        early_exaggeration: f64,
    ) -> Self {
        Self {
            perplexity,
            iterations,
            learning_rate,
            early_exaggeration,
            seed: None,
        }
    }
    /// Seed local initialization without changing the global random stream.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Return owned 2D coordinates in input row order, retaining row labels.
    /// Graph topology is not an input to this feature-space projection.
    pub fn project_matrix(&self, input: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let n = input.shape().0;
        if n < 2
            || !self.perplexity.is_finite()
            || self.perplexity < 1.0
            || self.perplexity > (n - 1) as f64
        {
            return Err(invalid(
                "t-SNE perplexity must be finite and in [1, number_of_rows - 1]",
            ));
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.early_exaggeration.is_finite()
            || self.early_exaggeration < 1.0
        {
            return Err(invalid(
                "t-SNE requires a finite positive learning rate and finite exaggeration >= 1",
            ));
        }
        let distances = squared_distances(input)?;
        let p = probabilities(&distances, n, self.perplexity)?;
        let mut rng = self
            .seed
            .map(fastrand::Rng::with_seed)
            .unwrap_or_else(fastrand::Rng::new);
        let mut data: Vec<_> = (0..n * 2).map(|_| (rng.f64() * 2.0 - 1.0) * 1e-4).collect();
        // Keep the phase transition even when an exaggerated phase stops early.
        for (iterations, exaggeration) in [
            (self.iterations.min(250), self.early_exaggeration),
            (self.iterations.saturating_sub(250), 1.0),
        ] {
            'phase: for _ in 0..iterations {
                let (energy, gradient) = objective(&data, &p, exaggeration)?;
                if gradient.iter().all(|&g| g == 0.0) {
                    break;
                }
                let mut step = self.learning_rate;
                let mut accepted = None;
                for _ in 0..64 {
                    let next: Vec<_> = data
                        .iter()
                        .zip(&gradient)
                        .map(|(x, g)| x - step * g)
                        .collect();
                    if next == data {
                        break 'phase;
                    }
                    if let Ok((candidate, _)) = objective(&next, &p, exaggeration) {
                        if candidate <= energy {
                            accepted = Some(next);
                            break;
                        }
                    }
                    step *= 0.5;
                }
                data = accepted.ok_or_else(|| {
                    invalid("t-SNE line search failed to find a finite nonincreasing step")
                })?;
            }
        }
        coordinates(input, data, "tsne")
    }
}
impl ProjectionEngine for TSNEProjection {
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
        "tsne"
    }
}

/// Match each conditional entropy, then symmetrize to a joint distribution of mass one.
fn probabilities(distances: &[f64], n: usize, perplexity: f64) -> GraphResult<Vec<f64>> {
    let mut conditional = vec![0.0; n * n];
    let target = perplexity.ln();
    for i in 0..n {
        let nearest = (0..n)
            .filter(|&j| j != i)
            .map(|j| distances[i * n + j])
            .fold(f64::INFINITY, f64::min);
        let farthest = (0..n)
            .filter(|&j| j != i)
            .map(|j| distances[i * n + j])
            .fold(0.0, f64::max);
        let ties = (0..n)
            .filter(|&j| j != i && distances[i * n + j] == nearest)
            .count();
        if target + 1e-8 < (ties as f64).ln() {
            return Err(invalid(
                "t-SNE perplexity is unattainable for tied nearest-neighbor distances",
            ));
        }
        if (target - ((n - 1) as f64).ln()).abs() <= 1e-8 {
            for j in 0..n {
                if i != j {
                    conditional[i * n + j] = 1.0 / (n - 1) as f64;
                }
            }
            continue;
        }
        if (target - (ties as f64).ln()).abs() <= 1e-8 {
            for j in 0..n {
                if j != i && distances[i * n + j] == nearest {
                    conditional[i * n + j] = 1.0 / ties as f64;
                }
            }
            continue;
        }
        let range = farthest - nearest;
        let mut lower = 0.0;
        let mut upper = f64::INFINITY;
        let mut beta = 1.0;
        let mut matched = false;
        for _ in 0..200 {
            let mut mass = 0.0;
            for j in 0..n {
                if i != j {
                    let weight = (-beta * ((distances[i * n + j] - nearest) / range)).exp();
                    conditional[i * n + j] = weight;
                    mass += weight;
                }
            }
            let mut entropy = 0.0;
            for j in 0..n {
                if i != j {
                    let p = conditional[i * n + j] / mass;
                    conditional[i * n + j] = p;
                    if p > 0.0 {
                        entropy -= p * p.ln();
                    }
                }
            }
            if (entropy - target).abs() <= 1e-8 {
                matched = true;
                break;
            }
            if entropy > target {
                lower = beta;
                beta = if upper.is_finite() {
                    (lower + upper) * 0.5
                } else {
                    beta * 2.0
                };
            } else {
                upper = beta;
                beta = (lower + upper) * 0.5;
            }
        }
        if !matched {
            return Err(invalid(
                "t-SNE perplexity calibration did not converge in 200 steps",
            ));
        }
    }
    let mut joint = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let p = (conditional[i * n + j] + conditional[j * n + i]) / (2.0 * n as f64);
            joint[i * n + j] = p;
            joint[j * n + i] = p;
        }
    }
    Ok(joint)
}

/// L = -exaggeration * sum(P_ij log(w_ij)) + log(sum(w_ij)), w=1/(1+r²).
/// At exaggeration one this differs from KL(P||Q) only by sum(P log P).
fn objective(data: &[f64], p: &[f64], exaggeration: f64) -> GraphResult<(f64, Vec<f64>)> {
    if data.iter().any(|x| !x.is_finite()) {
        return Err(invalid("t-SNE coordinates must be finite"));
    }
    let n = data.len() / 2;
    let mut weights = vec![0.0; n * n];
    let mut mass = 0.0;
    let mut attraction = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            let dx = data[2 * i] - data[2 * j];
            let dy = data[2 * i + 1] - data[2 * j + 1];
            let squared = dx * dx + dy * dy;
            if !squared.is_finite() {
                return Err(invalid("t-SNE coordinate distance overflow"));
            }
            let w = 1.0 / (1.0 + squared);
            weights[i * n + j] = w;
            mass += 2.0 * w;
            attraction += 2.0 * p[i * n + j] * squared.ln_1p();
        }
    }
    let energy = exaggeration * attraction + mass.ln();
    let mut gradient = vec![0.0; data.len()];
    for i in 0..n {
        for j in i + 1..n {
            let w = weights[i * n + j];
            let coefficient = 4.0 * (exaggeration * p[i * n + j] - w / mass) * w;
            for axis in 0..2 {
                let force = coefficient * (data[2 * i + axis] - data[2 * j + axis]);
                gradient[2 * i + axis] += force;
                gradient[2 * j + axis] -= force;
            }
        }
    }
    if !energy.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
        return Err(invalid("t-SNE objective or gradient overflow"));
    }
    Ok((energy, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibrated_joint_probabilities_are_symmetric_normalized_and_scale_invariant() {
        // For three points with distances (1,4,1), middle conditionals are uniform.
        let d = [0., 1., 4., 1., 0., 1., 4., 1., 0.];
        let p = probabilities(&d, 3, 2.).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(p[i * 3 + j], p[j * 3 + i]);
                assert!((p[i * 3 + j] - if i == j { 0.0 } else { 1.0 / 6.0 }).abs() < 1e-15);
            }
        }
        assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-15);
        assert!(probabilities(&d, 3, 1.).is_err()); // Tied middle row cannot have perplexity one.
                                                    // Analytic three-candidate distribution with entropy strictly between bounds.
        let d = [
            0., 1., 4., 9., 1., 0., 2., 8., 4., 2., 0., 3., 9., 8., 3., 0.,
        ];
        let first = probabilities(&d, 4, 1.7).unwrap();
        let scaled: Vec<_> = d.iter().map(|x| x * 1e100).collect();
        let second = probabilities(&scaled, 4, 1.7).unwrap();
        assert!((first.iter().sum::<f64>() - 1.).abs() < 1e-14);
        for (a, b) in first.iter().zip(second) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    #[test]
    fn calibrated_entropy_matches_known_conditional_distribution() {
        // Each row sees distances 1,2,4. At precision ln(2), conditional
        // probabilities are [8/13,4/13,1/13] in distance order.
        let weights = [8.0_f64 / 13.0, 4.0 / 13.0, 1.0 / 13.0];
        let perplexity = (-weights.iter().map(|p| p * p.ln()).sum::<f64>()).exp();
        let distances = [
            0., 1., 2., 4., 1., 0., 4., 2., 2., 4., 0., 1., 4., 2., 1., 0.,
        ];
        let p = probabilities(&distances, 4, perplexity).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let conditional = match distances[4 * i + j] {
                    0.0 => 0.0,
                    1.0 => weights[0],
                    2.0 => weights[1],
                    _ => weights[2],
                };
                assert!((p[4 * i + j] - conditional / 4.0).abs() < 1e-8);
            }
        }
    }
    #[test]
    fn optimizer_decreases_kl_without_exaggeration_and_handles_phase_transition() {
        let input =
            GraphMatrix::from_row_major_data(vec![0., 0., 1., 0., 0., 2., 3., 1.], 4, 2, None)
                .unwrap();
        let p = probabilities(&squared_distances(&input).unwrap(), 4, 1.7).unwrap();
        let before = TSNEProjection::new(1.7, 0, 100., 1.)
            .with_seed(42)
            .project_matrix(&input)
            .unwrap()
            .to_vec()
            .unwrap();
        let after = TSNEProjection::new(1.7, 100, 100., 1.)
            .with_seed(42)
            .project_matrix(&input)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(objective(&after, &p, 1.).unwrap().0 < objective(&before, &p, 1.).unwrap().0);
        let phase = TSNEProjection::new(1.7, 250, 100., 4.)
            .with_seed(42)
            .project_matrix(&input)
            .unwrap()
            .to_vec()
            .unwrap();
        let next = TSNEProjection::new(1.7, 251, 100., 4.)
            .with_seed(42)
            .project_matrix(&input)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(objective(&next, &p, 1.).unwrap().0 <= objective(&phase, &p, 1.).unwrap().0);
    }
    #[test]
    fn actual_kl_and_exaggerated_objective_have_correct_gradients() {
        let p = probabilities(
            &[
                0., 1., 4., 9., 1., 0., 2., 8., 4., 2., 0., 3., 9., 8., 3., 0.,
            ],
            4,
            1.7,
        )
        .unwrap();
        let data = [-0.4, 0.6, 0.8, 0.1, 0.3, -0.9, 1.2, 0.5];
        // Independent direct normalized Student-t formula.
        let reference = |x: &[f64], ex: f64| {
            let mut mass = 0.0;
            let mut loss = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        let weight = 1.0
                            / (1.0
                                + (x[2 * i] - x[2 * j]).powi(2)
                                + (x[2 * i + 1] - x[2 * j + 1]).powi(2));
                        mass += weight;
                        loss -= ex * p[i * 4 + j] * weight.ln();
                    }
                }
            }
            loss + mass.ln()
        };
        for ex in [1., 4., 12.] {
            let (energy, gradient) = objective(&data, &p, ex).unwrap();
            assert!((energy - reference(&data, ex)).abs() < 1e-13);
            for axis in 0..data.len() {
                let mut plus = data;
                let mut minus = data;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let want = (reference(&plus, ex) - reference(&minus, ex)) / 2e-6;
                assert!((gradient[axis] - want).abs() < 1e-8);
            }
        }
    }
}
