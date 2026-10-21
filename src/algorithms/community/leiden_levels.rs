//! Bounded modularity Leiden: fast moves, constrained stochastic singleton
//! refinement, and refined quotient graphs carrying the unrefined partition.
use super::modularity_levels::{Level, Partition};
use crate::algorithms::Context;
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const GAIN_EPSILON: f64 = 1e-12;

fn compact(mut labels: Vec<usize>, moves: usize, iterations: usize) -> Partition {
    let mut remap = vec![usize::MAX; labels.len()];
    let mut communities = 0;
    for c in &mut labels {
        if remap[*c] == usize::MAX {
            remap[*c] = communities;
            communities += 1;
        }
        *c = remap[*c];
    }
    Partition {
        labels,
        communities,
        moves,
        iterations,
    }
}

pub(super) fn fast_moves(
    level: &Level,
    initial: &[usize],
    gamma: f64,
    max_iter: usize,
    rng: &mut fastrand::Rng,
    ctx: &Context,
) -> Result<Partition> {
    level.modularity(initial, gamma)?; // Validate inherited labels before indexing.
    let mut labels = initial.to_vec();
    let mut totals = vec![0.; level.n];
    let mut sizes = vec![0; level.n];
    for (u, &c) in labels.iter().enumerate() {
        totals[c] += level.degrees[u];
        sizes[c] += 1;
    }
    let mut empty: BTreeSet<_> = (0..level.n).filter(|&c| sizes[c] == 0).collect();
    let mut order: Vec<_> = (0..level.n).collect();
    rng.shuffle(&mut order);
    let mut queue: VecDeque<_> = order.into();
    let mut queued = vec![true; level.n];
    let budget = level
        .n
        .checked_mul(max_iter)
        .ok_or_else(|| anyhow!("Leiden visit budget overflow"))?;
    let mut visits = 0;
    let mut moves = 0;
    let m = level.mass as f64;
    while visits < budget {
        if ctx.is_cancelled() {
            return Err(anyhow!("Leiden cancelled"));
        }
        let Some(u) = queue.pop_front() else {
            break;
        };
        queued[u] = false;
        visits += 1;
        if m == 0. {
            continue;
        }
        let old = labels[u];
        let degree = level.degrees[u];
        let mut connections: BTreeMap<usize, f64> = BTreeMap::new();
        for &(v, w) in level.adjacency(u) {
            *connections.entry(labels[v]).or_default() += w;
        }
        // A coarse node must be able to leave its inherited group for a new singleton.
        if sizes[old] > 1 {
            if let Some(&free) = empty.first() {
                connections.entry(free).or_insert(0.);
            }
        }
        let old_inside = connections.get(&old).copied().unwrap_or(0.);
        let mut best = old;
        let mut best_gain = 0.;
        for (&candidate, &inside) in &connections {
            if candidate == old {
                continue;
            }
            let gain = (inside - old_inside) / m
                + gamma * ((degree / (2. * m)) * ((totals[old] - totals[candidate] - degree) / m));
            if !gain.is_finite() {
                return Err(anyhow!("Leiden gain is not finite"));
            }
            if gain > best_gain + GAIN_EPSILON {
                best = candidate;
                best_gain = gain;
            }
        }
        if best != old {
            totals[old] -= degree;
            totals[best] += degree;
            sizes[old] -= 1;
            sizes[best] += 1;
            if sizes[old] == 0 {
                empty.insert(old);
            }
            empty.remove(&best);
            labels[u] = best;
            moves += 1;
            for &(v, _) in level.adjacency(u) {
                if labels[v] != best && !queued[v] {
                    queue.push_back(v);
                    queued[v] = true;
                }
            }
        }
    }
    // Bounded visits can stop before the queue converges; split any disconnected
    // remainder before publishing/aggregating to preserve connected output.
    let mut connected = vec![usize::MAX; level.n];
    let mut stack = Vec::new();
    let mut next = 0;
    for u in 0..level.n {
        if connected[u] != usize::MAX {
            continue;
        }
        connected[u] = next;
        stack.push(u);
        while let Some(v) = stack.pop() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Leiden cancelled"));
            }
            for &(w, _) in level.adjacency(v) {
                if labels[w] == labels[u] && connected[w] == usize::MAX {
                    connected[w] = next;
                    stack.push(w);
                }
            }
        }
        next += 1;
    }
    Ok(compact(connected, moves, visits))
}

// Degree-weighted well-connectedness for modularity, compared in normalized
// units to avoid overflowing gamma * degree products.
fn eligible(cut: f64, degree: f64, parent_degree: f64, m: f64, gamma: f64) -> bool {
    cut / m >= gamma * ((degree / (2. * m)) * ((parent_degree - degree) / m))
}

pub(super) fn refine(
    level: &Level,
    parent: &Partition,
    gamma: f64,
    theta: f64,
    rng: &mut fastrand::Rng,
    ctx: &Context,
) -> Result<Partition> {
    level.modularity(&parent.labels, gamma)?;
    if !theta.is_finite() || theta <= 0. {
        return Err(anyhow!("Leiden theta must be finite and positive"));
    }
    let mut labels = level.singleton_labels();
    let mut sizes = vec![1; level.n];
    let mut totals = level.degrees.clone();
    let mut parent_totals = vec![0.; level.n];
    let mut cut = vec![0.; level.n];
    for u in 0..level.n {
        parent_totals[parent.labels[u]] += level.degrees[u];
        for &(v, w) in level.adjacency(u) {
            if parent.labels[u] == parent.labels[v] {
                cut[u] += w;
            }
        }
    }
    if level.mass == 0 {
        return Ok(compact(labels, 0, 0));
    }
    let m = level.mass as f64;
    let mut order: Vec<_> = (0..level.n).collect();
    rng.shuffle(&mut order);
    let mut merges = 0;
    for u in order {
        if ctx.is_cancelled() {
            return Err(anyhow!("Leiden cancelled"));
        }
        let own = labels[u];
        let parent_degree = parent_totals[parent.labels[u]];
        if sizes[own] != 1 || !eligible(cut[own], totals[own], parent_degree, m, gamma) {
            continue;
        }
        let mut connections: BTreeMap<usize, f64> = BTreeMap::new();
        for &(v, w) in level.adjacency(u) {
            if parent.labels[v] == parent.labels[u] {
                *connections.entry(labels[v]).or_default() += w;
            }
        }
        // Staying is a zero-gain refinement choice; merges must have a real
        // connection, remain inside the parent, and never decrease the objective.
        let mut candidates = vec![(own, 0., 0.)];
        for (&candidate, &inside) in &connections {
            if candidate == own
                || !eligible(cut[candidate], totals[candidate], parent_degree, m, gamma)
            {
                continue;
            }
            let gain =
                inside / m - gamma * ((level.degrees[u] / (2. * m)) * (totals[candidate] / m));
            if !gain.is_finite() {
                return Err(anyhow!("Leiden refinement gain is not finite"));
            }
            if gain >= 0. {
                candidates.push((candidate, gain, inside));
            }
        }
        let max_gain = candidates
            .iter()
            .fold(0_f64, |best, (_, gain, _)| best.max(*gain));
        let weights: Vec<_> = candidates
            .iter()
            .map(|(_, gain, _)| ((gain - max_gain) / theta).exp())
            .collect();
        let mut draw = rng.f64() * weights.iter().sum::<f64>();
        let mut choice = weights.iter().rposition(|&weight| weight > 0.).unwrap_or(0);
        for (i, &weight) in weights.iter().enumerate() {
            if draw < weight {
                choice = i;
                break;
            }
            draw -= weight;
        }
        let (target, _, inside) = candidates[choice];
        if target != own {
            cut[target] = (cut[target] + cut[own] - 2. * inside).max(0.);
            cut[own] = 0.;
            totals[target] += totals[own];
            totals[own] = 0.;
            sizes[target] += 1;
            sizes[own] = 0;
            labels[u] = target;
            merges += 1;
        }
    }
    Ok(compact(labels, merges, level.n))
}

pub(super) fn inherited_partition(parent: &Partition, refined: &Partition) -> Result<Vec<usize>> {
    if parent.labels.len() != refined.labels.len() {
        return Err(anyhow!("Leiden refinement length mismatch"));
    }
    let mut inherited = vec![usize::MAX; refined.communities];
    for (&p, &r) in parent.labels.iter().zip(&refined.labels) {
        let entry = inherited
            .get_mut(r)
            .ok_or_else(|| anyhow!("invalid Leiden refined label"))?;
        if *entry == usize::MAX {
            *entry = p;
        } else if *entry != p {
            return Err(anyhow!("Leiden refinement crossed its parent partition"));
        }
    }
    if inherited.contains(&usize::MAX) {
        return Err(anyhow!("empty Leiden refined group"));
    }
    Ok(inherited)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected(level: &Level, labels: &[usize]) {
        for root in 0..level.n {
            let mut seen = vec![false; level.n];
            let mut stack = vec![root];
            seen[root] = true;
            while let Some(u) = stack.pop() {
                for &(v, _) in level.adjacency(u) {
                    if labels[v] == labels[root] && !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
            for v in 0..level.n {
                if labels[v] == labels[root] {
                    assert!(seen[v]);
                }
            }
        }
    }

    #[test]
    fn refinement_splits_a_connected_parent_and_carries_it_into_the_quotient() {
        let level = Level::new(4, [(0, 1, 10), (2, 3, 10), (1, 2, 1)]).unwrap();
        let parent = Partition {
            labels: vec![0, 0, 0, 0],
            communities: 1,
            moves: 0,
            iterations: 0,
        };
        for seed in 0..8 {
            let refined = refine(
                &level,
                &parent,
                1.,
                1e-6,
                &mut fastrand::Rng::with_seed(seed),
                &Context::new(),
            )
            .unwrap();
            assert_eq!(refined.labels, vec![0, 0, 1, 1]);
            let inherited = inherited_partition(&parent, &refined).unwrap();
            assert_eq!(inherited, vec![0, 0]);
            let coarse = level.aggregate(&refined).unwrap();
            assert!(
                (coarse.modularity(&inherited, 1.).unwrap()
                    - level.modularity(&parent.labels, 1.).unwrap())
                .abs()
                    < 1e-12
            );
            let moved = fast_moves(
                &coarse,
                &inherited,
                1.,
                10,
                &mut fastrand::Rng::with_seed(seed),
                &Context::new(),
            )
            .unwrap();
            // Splitting the inherited community needs the empty/singleton candidate.
            assert_eq!(moved.communities, 2);
            assert!(coarse.modularity(&moved.labels, 1.).unwrap() > 0.4);
        }
    }

    #[test]
    fn refinement_respects_parent_boundaries_objective_and_connectedness() {
        let level = Level::new(
            6,
            [
                (0, 0, 3),
                (0, 1, 4),
                (1, 2, 2),
                (2, 3, 1),
                (3, 4, 5),
                (4, 4, 2),
            ],
        )
        .unwrap();
        for parent_labels in [vec![0, 0, 0, 1, 1, 2], vec![0, 0, 0, 0, 0, 0]] {
            let parent = compact(parent_labels, 0, 0);
            for gamma in [0.25, 1., 3.] {
                for theta in [1e-300, 0.01, 1e100] {
                    for seed in 0..12 {
                        let refined = refine(
                            &level,
                            &parent,
                            gamma,
                            theta,
                            &mut fastrand::Rng::with_seed(seed),
                            &Context::new(),
                        )
                        .unwrap();
                        inherited_partition(&parent, &refined).unwrap();
                        connected(&level, &refined.labels);
                        assert!(
                            level.modularity(&refined.labels, gamma).unwrap()
                                >= level.modularity(&level.singleton_labels(), gamma).unwrap()
                                    - 1e-12
                        );
                        assert_ne!(refined.labels[5], refined.labels[0]); // Isolate never merges.
                    }
                }
            }
        }
    }

    #[test]
    fn degree_weighted_eligibility_keeps_weakly_attached_loop_nodes_separate() {
        let level = Level::new(2, [(0, 0, 100), (1, 1, 100), (0, 1, 1)]).unwrap();
        let parent = Partition {
            labels: vec![0, 0],
            communities: 1,
            moves: 0,
            iterations: 0,
        };
        assert!(!eligible(1., 201., 402., 201., 1.));
        let result = refine(
            &level,
            &parent,
            1.,
            0.01,
            &mut fastrand::Rng::with_seed(1),
            &Context::new(),
        )
        .unwrap();
        assert_eq!(result.communities, 2);
    }

    #[test]
    fn visit_budget_and_internal_validation_reject_bad_inputs() {
        let level = Level::new(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap();
        let ctx = Context::new();
        let parent = fast_moves(
            &level,
            &[0, 0, 0, 0],
            1.,
            1,
            &mut fastrand::Rng::with_seed(9),
            &ctx,
        )
        .unwrap();
        assert!(parent.iterations <= 4);
        connected(&level, &parent.labels);
        assert!(fast_moves(
            &level,
            &[0, 1, 2, 3],
            1.,
            usize::MAX,
            &mut fastrand::Rng::with_seed(9),
            &ctx
        )
        .is_err());
        let bad = Partition {
            labels: vec![0, 0, 0, 0],
            communities: 1,
            moves: 0,
            iterations: 0,
        };
        let crossing = Partition {
            labels: vec![0, 0, 0, 0],
            communities: 1,
            moves: 0,
            iterations: 0,
        };
        let split = Partition {
            labels: vec![0, 0, 1, 1],
            communities: 2,
            moves: 0,
            iterations: 0,
        };
        assert!(inherited_partition(&split, &crossing).is_err());
        assert!(refine(&level, &bad, 1., 0., &mut fastrand::Rng::with_seed(9), &ctx).is_err());
    }
}
