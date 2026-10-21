//! Internal undirected quotient graphs. Multiplicity is internal edge mass, not
//! a public edge-weight attribute; self-loops preserve within-community mass.
use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

use crate::algorithms::Context;

const MOVE_EPSILON: f64 = 1e-6;

pub(super) struct Level {
    pub(super) n: usize,
    edges: Vec<(usize, usize, u64)>,
    offsets: Vec<usize>,
    neighbors: Vec<(usize, f64)>,
    pub(super) degrees: Vec<f64>,
    pub(super) mass: u64,
}

pub(super) struct Partition {
    pub labels: Vec<usize>,
    pub communities: usize,
    pub moves: usize,
    pub iterations: usize,
}

impl Level {
    pub fn new(n: usize, edges: impl IntoIterator<Item = (usize, usize, u64)>) -> Result<Self> {
        let mut pairs: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        let mut mass = 0_u64;
        for (u, v, weight) in edges {
            if u >= n || v >= n || weight == 0 {
                return Err(anyhow!("invalid community quotient edge"));
            }
            mass = mass
                .checked_add(weight)
                .ok_or_else(|| anyhow!("community edge mass overflow"))?;
            let pair = if u <= v { (u, v) } else { (v, u) };
            let entry = pairs.entry(pair).or_default();
            *entry = entry
                .checked_add(weight)
                .ok_or_else(|| anyhow!("community parallel mass overflow"))?;
        }
        let edges: Vec<_> = pairs.into_iter().map(|((u, v), w)| (u, v, w)).collect();
        let mut offsets = vec![0; n + 1];
        let mut degrees = vec![0.; n];
        for &(u, v, w) in &edges {
            degrees[u] += w as f64;
            degrees[v] += w as f64;
            if u != v {
                offsets[u + 1] += 1;
                offsets[v + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut neighbors = vec![(0, 0.); offsets[n]];
        let mut cursor = offsets[..n].to_vec();
        for &(u, v, w) in &edges {
            if u != v {
                neighbors[cursor[u]] = (v, w as f64);
                cursor[u] += 1;
                neighbors[cursor[v]] = (u, w as f64);
                cursor[v] += 1;
            }
        }
        Ok(Self {
            n,
            edges,
            offsets,
            neighbors,
            degrees,
            mass,
        })
    }

    pub fn adjacency(&self, node: usize) -> &[(usize, f64)] {
        &self.neighbors[self.offsets[node]..self.offsets[node + 1]]
    }

    pub fn singleton_labels(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    pub fn modularity(&self, labels: &[usize], gamma: f64) -> Result<f64> {
        if labels.len() != self.n || labels.iter().any(|&c| c >= self.n) {
            return Err(anyhow!("invalid community quotient labels"));
        }
        if self.mass == 0 {
            return Ok(0.);
        }
        let m = self.mass as f64;
        let mut internal = vec![0.; self.n];
        let mut totals = vec![0.; self.n];
        for (u, &c) in labels.iter().enumerate() {
            totals[c] += self.degrees[u];
        }
        for &(u, v, w) in &self.edges {
            if labels[u] == labels[v] {
                internal[labels[u]] += w as f64;
            }
        }
        let q: f64 = internal
            .iter()
            .zip(totals)
            .map(|(&inside, tot)| inside / m - gamma * (tot / (2. * m)).powi(2))
            .sum();
        if !q.is_finite() {
            return Err(anyhow!("community modularity is not finite"));
        }
        Ok(q)
    }

    pub fn local_moves(&self, gamma: f64, max_iter: usize, ctx: &Context) -> Result<Partition> {
        let mut labels = self.singleton_labels();
        let mut totals = self.degrees.clone();
        let mut moves = 0;
        let mut iterations = 0;
        if self.mass > 0 {
            let m = self.mass as f64;
            for _ in 0..max_iter {
                let mut changed = false;
                for u in 0..self.n {
                    if ctx.is_cancelled() {
                        return Err(anyhow!("louvain cancelled"));
                    }
                    let old = labels[u];
                    let degree = self.degrees[u];
                    let mut connections: BTreeMap<usize, f64> = BTreeMap::new();
                    for &(v, w) in &self.neighbors[self.offsets[u]..self.offsets[u + 1]] {
                        *connections.entry(labels[v]).or_default() += w;
                    }
                    let old_inside = connections.get(&old).copied().unwrap_or(0.);
                    let mut best = old;
                    let mut best_gain = 0.;
                    for (&candidate, &inside) in &connections {
                        if candidate == old {
                            continue;
                        }
                        let gain = (inside - old_inside) / m
                            + gamma
                                * ((degree / (2. * m))
                                    * ((totals[old] - totals[candidate] - degree) / m));
                        if !gain.is_finite() {
                            return Err(anyhow!("Louvain modularity gain is not finite"));
                        }
                        if gain > best_gain + MOVE_EPSILON {
                            best = candidate;
                            best_gain = gain;
                        }
                    }
                    if best != old {
                        totals[old] -= degree;
                        totals[best] += degree;
                        labels[u] = best;
                        moves += 1;
                        changed = true;
                    }
                }
                iterations += 1;
                if !changed {
                    break;
                }
            }
        }
        // Canonical level labels follow first occurrence in dense node order.
        let mut remap = vec![usize::MAX; self.n];
        let mut communities = 0;
        for label in &mut labels {
            if remap[*label] == usize::MAX {
                remap[*label] = communities;
                communities += 1;
            }
            *label = remap[*label];
        }
        Ok(Partition {
            labels,
            communities,
            moves,
            iterations,
        })
    }

    pub fn aggregate(&self, partition: &Partition) -> Result<Self> {
        if partition.labels.len() != self.n
            || partition.labels.iter().any(|&c| c >= partition.communities)
        {
            return Err(anyhow!("invalid community aggregation partition"));
        }
        Self::new(
            partition.communities,
            self.edges
                .iter()
                .map(|&(u, v, w)| (partition.labels[u], partition.labels[v], w)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(n: usize, edges: &[(usize, usize, u64)], labels: &[usize], gamma: f64) -> f64 {
        let mut a = vec![vec![0.; n]; n];
        let mut degrees = vec![0.; n];
        let mut twice_m = 0.;
        for &(u, v, w) in edges {
            a[u][v] += w as f64;
            a[v][u] += w as f64;
            degrees[u] += w as f64;
            degrees[v] += w as f64;
            twice_m += 2. * w as f64;
        }
        if twice_m == 0. {
            return 0.;
        }
        let mut q = 0.;
        for i in 0..n {
            for j in 0..n {
                if labels[i] == labels[j] {
                    q += a[i][j] - gamma * degrees[i] * degrees[j] / twice_m;
                }
            }
        }
        q / twice_m
    }

    #[test]
    fn quotient_preserves_original_objective_for_every_small_coarse_partition() {
        let edges = [
            (0, 0, 2),
            (0, 1, 1),
            (1, 0, 3),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 3),
            (7, 7, 1),
            (7, 0, 2),
        ];
        let level = Level::new(9, edges).unwrap(); // Original vertex 8 is isolated.
        let membership = vec![0, 0, 1, 1, 2, 2, 3, 3, 4];
        let partition = Partition {
            labels: membership.clone(),
            communities: 5,
            moves: 0,
            iterations: 0,
        };
        let coarse = level.aggregate(&partition).unwrap();
        assert_eq!(level.mass, coarse.mass);
        for code in 0_usize..243 {
            let mut code = code;
            let labels: Vec<_> = (0..5)
                .map(|_| {
                    let c = code % 3;
                    code /= 3;
                    c
                })
                .collect();
            let lifted: Vec<_> = membership.iter().map(|&c| labels[c]).collect();
            for gamma in [0.25, 1., 3.] {
                let expected = reference(9, &edges, &lifted, gamma);
                assert!((coarse.modularity(&labels, gamma).unwrap() - expected).abs() < 1e-12);
                assert!((level.modularity(&lifted, gamma).unwrap() - expected).abs() < 1e-12);
            }
        }
        let second = coarse
            .aggregate(&Partition {
                labels: vec![0, 0, 1, 1, 2],
                communities: 3,
                moves: 0,
                iterations: 0,
            })
            .unwrap();
        assert_eq!(second.mass, level.mass);
        assert!(
            (second.modularity(&[0, 1, 2], 1.).unwrap()
                - reference(9, &edges, &[0, 0, 0, 0, 1, 1, 1, 1, 2], 1.))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn invalid_mass_partitions_and_nonfinite_objectives_reject() {
        assert!(Level::new(1, [(0, 1, 1)]).is_err());
        assert!(Level::new(1, [(0, 0, 0)]).is_err());
        assert!(Level::new(1, [(0, 0, u64::MAX), (0, 0, 1)]).is_err());
        let level = Level::new(2, [(0, 1, 1)]).unwrap();
        assert!(level.modularity(&[0], 1.).is_err());
        assert!(level.modularity(&[0, 1], f64::NAN).is_err());
        assert!(level
            .aggregate(&Partition {
                labels: vec![0, 2],
                communities: 2,
                moves: 0,
                iterations: 0
            })
            .is_err());
    }
}
