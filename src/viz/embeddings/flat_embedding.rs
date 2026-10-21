//! Bounded graph layout in a disk, with explicit native objective gradients.

use super::row_node_ids;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::streaming::data_source::Position;
use std::collections::HashMap;

/// Parameters for a seeded projected-gradient graph layout.
#[derive(Debug, Clone)]
pub struct FlatEmbedConfig {
    pub iterations: usize,
    pub learning_rate: f64,
    pub edge_cohesion_weight: f64,
    pub repulsion_weight: f64,
    pub repulsion_power: f64,
    pub spread_weight: f64,
    pub radius_clip: f64,
    pub seed: u64,
}
impl Default for FlatEmbedConfig {
    fn default() -> Self {
        Self {
            iterations: 800,
            learning_rate: 0.03,
            edge_cohesion_weight: 1.0,
            repulsion_weight: 0.1,
            repulsion_power: 1.5,
            spread_weight: 0.05,
            radius_clip: 0.98,
            seed: 0,
        }
    }
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
impl FlatEmbedConfig {
    fn validate(&self) -> GraphResult<()> {
        for (name, value) in [
            ("learning_rate", self.learning_rate),
            ("repulsion_power", self.repulsion_power),
            ("radius_clip", self.radius_clip),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!("Flat {name} must be finite and positive")));
            }
        }
        for (name, value) in [
            ("edge_cohesion_weight", self.edge_cohesion_weight),
            ("repulsion_weight", self.repulsion_weight),
            ("spread_weight", self.spread_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(format!(
                    "Flat {name} must be finite and nonnegative"
                )));
            }
        }
        Ok(())
    }
}

/// Layout rows against actual active graph topology; unlabeled rows use ascending IDs.
/// Feature values do not enter this graph-layout objective. The matrix supplies
/// row identity/shape, not a promise to preserve high-dimensional distances.
pub fn compute_flat_embedding(
    embedding: &GraphMatrix,
    graph: &Graph,
    config: &FlatEmbedConfig,
) -> GraphResult<Vec<Position>> {
    config.validate()?;
    let nodes = row_node_ids(embedding, graph)?;
    let index: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();
    let mut ids = graph.space().edge_ids();
    ids.sort_unstable();
    let mut edges = Vec::with_capacity(ids.len());
    for edge in ids {
        let (u, v) = graph.edge_endpoints(edge)?;
        let u = index
            .get(&u)
            .copied()
            .ok_or_else(|| invalid(format!("Edge {edge} endpoint outside embedding")))?;
        let v = index
            .get(&v)
            .copied()
            .ok_or_else(|| invalid(format!("Edge {edge} endpoint outside embedding")))?;
        edges.push((u, v)); // Each real instance, including loops; never truncate.
    }
    layout(nodes.len(), &edges, config)
}

/// Owned graph-aware matrix result in input row order, with validated ID labels.
pub fn compute_flat_embedding_matrix(
    embedding: &GraphMatrix,
    graph: &Graph,
    config: &FlatEmbedConfig,
) -> GraphResult<GraphMatrix> {
    let nodes = row_node_ids(embedding, graph)?;
    let positions = compute_flat_embedding(embedding, graph, config)?;
    let mut matrix = positions_matrix(positions)?;
    matrix.set_row_labels(Some(
        nodes.iter().map(|node| format!("node_{node}")).collect(),
    ));
    Ok(matrix)
}

/// Graph-free layout requires cohesion to be disabled; a matrix cannot supply edges.
pub fn compute_flat_embedding_from_matrix(
    embedding: &GraphMatrix,
    config: &FlatEmbedConfig,
) -> GraphResult<GraphMatrix> {
    config.validate()?;
    if embedding.shape().0 > 1 && config.edge_cohesion_weight != 0.0 {
        return Err(invalid("Matrix-only flat layout has no topology; pass a graph or explicitly set edge_cohesion_weight=0"));
    }
    let mut result = positions_matrix(layout(embedding.shape().0, &[], config)?)?;
    result.set_row_labels(embedding.row_labels().map(<[String]>::to_vec));
    Ok(result)
}
fn positions_matrix(positions: Vec<Position>) -> GraphResult<GraphMatrix> {
    let rows = positions.len();
    let mut result = GraphMatrix::from_row_major_data(
        positions.into_iter().flat_map(|p| [p.x, p.y]).collect(),
        rows,
        2,
        None,
    )?;
    result.set_column_names(vec!["flat_x".into(), "flat_y".into()]);
    Ok(result)
}

fn clip(positions: &mut [[f64; 2]], radius: f64) {
    for point in positions {
        let scale = point[0].abs().max(point[1].abs());
        if scale == 0.0 {
            continue;
        }
        let norm = (point[0] / scale).hypot(point[1] / scale);
        if scale > radius / norm {
            point[0] = (point[0] / scale) / norm * radius;
            point[1] = (point[1] / scale) / norm * radius;
        }
    }
}

/// E = wc*mean_edges(||u-v||²) + wr*mean_pairs((||i-j||²+1e-6)^(-p/2))
///     - ws*det(population covariance). Loops contribute zero to the edge mean.
fn energy_gradient(
    points: &[[f64; 2]],
    edges: &[(usize, usize)],
    config: &FlatEmbedConfig,
) -> GraphResult<(f64, Vec<[f64; 2]>)> {
    let n = points.len();
    let mut gradient = vec![[0.0; 2]; n];
    let mut energy = 0.0;
    let cohesion = if edges.is_empty() {
        0.0
    } else {
        config.edge_cohesion_weight / edges.len() as f64
    };
    for &(u, v) in edges {
        if u >= n || v >= n {
            return Err(invalid("Flat edge index outside position rows"));
        }
        if cohesion == 0.0 || u == v {
            continue;
        }
        for axis in 0..2 {
            let delta = points[u][axis] - points[v][axis];
            energy += cohesion * delta * delta;
            gradient[u][axis] += 2.0 * cohesion * delta;
            gradient[v][axis] -= 2.0 * cohesion * delta;
        }
    }
    if config.repulsion_weight > 0.0 && n > 1 {
        let pairs = n
            .checked_mul(n - 1)
            .map(|p| p / 2)
            .ok_or_else(|| invalid("Flat pair count overflow"))?;
        let weight = config.repulsion_weight / pairs as f64;
        for u in 0..n {
            for v in u + 1..n {
                let dx = points[u][0] - points[v][0];
                let dy = points[u][1] - points[v][1];
                let squared = dx * dx + dy * dy + 1e-6;
                let term = weight * squared.powf(-config.repulsion_power / 2.0);
                energy += term;
                let factor = -config.repulsion_power * term / squared;
                for (axis, delta) in [dx, dy].into_iter().enumerate() {
                    gradient[u][axis] += factor * delta;
                    gradient[v][axis] -= factor * delta;
                }
            }
        }
    }
    if config.spread_weight > 0.0 && n > 0 {
        let mut mean = [0.0; 2];
        for (i, point) in points.iter().enumerate() {
            for axis in 0..2 {
                mean[axis] += (point[axis] - mean[axis]) / (i + 1) as f64;
            }
        }
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xy = 0.0;
        for point in points {
            let x = point[0] - mean[0];
            let y = point[1] - mean[1];
            xx += x * x / n as f64;
            yy += y * y / n as f64;
            xy += x * y / n as f64;
        }
        energy -= config.spread_weight * (xx * yy - xy * xy);
        let factor = -2.0 * config.spread_weight / n as f64;
        for (point, grad) in points.iter().zip(&mut gradient) {
            let x = point[0] - mean[0];
            let y = point[1] - mean[1];
            grad[0] += factor * (yy * x - xy * y);
            grad[1] += factor * (xx * y - xy * x);
        }
    }
    if !energy.is_finite() || gradient.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("Flat objective/gradient exceeds finite f64 range"));
    }
    Ok((energy, gradient))
}

fn layout(
    n: usize,
    edges: &[(usize, usize)],
    config: &FlatEmbedConfig,
) -> GraphResult<Vec<Position>> {
    n.checked_mul(2)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| invalid("Flat position size overflow"))?;
    let mut rng = fastrand::Rng::with_seed(config.seed);
    let mut points: Vec<_> = (0..n)
        .map(|_| [rng.f64() * 0.3 - 0.15, rng.f64() * 0.3 - 0.15])
        .collect();
    clip(&mut points, config.radius_clip);
    for iteration in 0..config.iterations {
        let (energy, gradient) = energy_gradient(&points, edges, config)?;
        if gradient.iter().flatten().all(|&g| g == 0.0) {
            break;
        }
        let mut step = config.learning_rate;
        let mut accepted = None;
        for _ in 0..32 {
            let mut candidate: Vec<_> = points
                .iter()
                .zip(&gradient)
                .map(|(p, g)| [p[0] - step * g[0], p[1] - step * g[1]])
                .collect();
            if candidate.iter().flatten().all(|x| x.is_finite()) {
                clip(&mut candidate, config.radius_clip);
                if let Ok((next_energy, _)) = energy_gradient(&candidate, edges, config) {
                    if next_energy <= energy {
                        accepted = Some(candidate);
                        break;
                    }
                }
            }
            step *= 0.5;
        }
        let next = accepted.ok_or_else(|| {
            invalid(format!(
                "Flat layout failed to find descent at iteration {iteration}"
            ))
        })?;
        if next == points {
            break;
        }
        points = next;
    }
    Ok(points.into_iter().map(|[x, y]| Position { x, y }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_gradients_match_independent_finite_differences() {
        let points = [[0.15, -0.1], [-0.2, 0.25], [0.35, 0.3]];
        let edges = [(0, 1), (0, 1), (2, 2), (1, 2)];
        for (cohesion, repulsion, spread) in [
            (1.0, 0.0, 0.0),
            (0.0, 0.4, 0.0),
            (0.0, 0.0, 0.8),
            (1.0, 0.4, 0.8),
        ] {
            let config = FlatEmbedConfig {
                edge_cohesion_weight: cohesion,
                repulsion_weight: repulsion,
                spread_weight: spread,
                ..Default::default()
            };
            let reference = |p: &[[f64; 2]]| {
                let cohesion = edges
                    .iter()
                    .map(|&(u, v)| (p[u][0] - p[v][0]).powi(2) + (p[u][1] - p[v][1]).powi(2))
                    .sum::<f64>()
                    / edges.len() as f64;
                let repulsion = (0..3)
                    .flat_map(|u| (u + 1..3).map(move |v| (u, v)))
                    .map(|(u, v)| {
                        ((p[u][0] - p[v][0]).powi(2) + (p[u][1] - p[v][1]).powi(2) + 1e-6)
                            .powf(-config.repulsion_power / 2.0)
                    })
                    .sum::<f64>()
                    / 3.0;
                let mx = p.iter().map(|v| v[0]).sum::<f64>() / 3.0;
                let my = p.iter().map(|v| v[1]).sum::<f64>() / 3.0;
                let xx = p.iter().map(|v| (v[0] - mx).powi(2)).sum::<f64>() / 3.0;
                let yy = p.iter().map(|v| (v[1] - my).powi(2)).sum::<f64>() / 3.0;
                let xy = p.iter().map(|v| (v[0] - mx) * (v[1] - my)).sum::<f64>() / 3.0;
                config.edge_cohesion_weight * cohesion + config.repulsion_weight * repulsion
                    - config.spread_weight * (xx * yy - xy * xy)
            };
            let (energy, gradient) = energy_gradient(&points, &edges, &config).unwrap();
            assert!((energy - reference(&points)).abs() < 1e-12);
            for (row, grad) in gradient.iter().enumerate() {
                for (axis, &got) in grad.iter().enumerate() {
                    let mut plus = points;
                    let mut minus = points;
                    plus[row][axis] += 1e-6;
                    minus[row][axis] -= 1e-6;
                    let want = (reference(&plus) - reference(&minus)) / 2e-6;
                    assert!((got - want).abs() < 1e-5);
                }
            }
        }
    }
    #[test]
    fn circle_projection_handles_extreme_finite_coordinates() {
        let mut points = [[f64::MAX, f64::MAX], [0.0, 0.0]];
        clip(&mut points, 0.1);
        assert!((points[0][0].hypot(points[0][1]) - 0.1).abs() < 1e-15);
        assert_eq!(points[1], [0.0, 0.0]);
    }
}
