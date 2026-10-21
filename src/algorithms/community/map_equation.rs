//! Recorded uniform-node teleportation and the two-level map equation.
use crate::algorithms::Context;
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

pub(super) struct Walk {
    n: usize,
    offsets: Vec<usize>,
    arcs: Vec<(usize, f64)>, // Row probabilities, retaining parallel multiplicity.
}
pub(super) struct Flow {
    pub visits: Vec<f64>,
    pub jumps: Vec<f64>,
    outgoing: Vec<Vec<(usize, f64)>>,
    incoming: Vec<Vec<(usize, f64)>>,
    pub iterations: usize,
    pub residual: f64,
}

fn xlog(x: f64) -> f64 {
    if x <= 0. {
        0.
    } else {
        x * x.log2()
    }
}
fn term(exit: f64, mass: f64) -> f64 {
    -2. * xlog(exit) + xlog(exit + mass)
}
fn nonnegative(x: f64) -> Result<f64> {
    if !x.is_finite() || x < -1e-11 {
        return Err(anyhow!("invalid map-equation flow"));
    }
    Ok(x.max(0.))
}

impl Walk {
    pub fn new(
        n: usize,
        edges: impl IntoIterator<Item = (usize, usize)>,
        directed: bool,
    ) -> Result<Self> {
        let mut pairs: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (u, v) in edges {
            if u >= n || v >= n {
                return Err(anyhow!("map-equation endpoint outside selection"));
            }
            let entry = pairs.entry((u, v)).or_default();
            *entry = entry
                .checked_add(1)
                .ok_or_else(|| anyhow!("Infomap edge count overflow"))?;
            if !directed {
                let entry = pairs.entry((v, u)).or_default();
                *entry = entry
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Infomap edge count overflow"))?;
            } // Loop contributes twice.
        }
        let mut degree = vec![0.; n];
        let mut offsets = vec![0; n + 1];
        for (&(u, _), &count) in &pairs {
            degree[u] += count as f64;
            offsets[u + 1] += 1;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let arcs = pairs
            .into_iter()
            .map(|((u, v), count)| (v, count as f64 / degree[u]))
            .collect();
        Ok(Self { n, offsets, arcs })
    }
    fn row(&self, u: usize) -> &[(usize, f64)] {
        &self.arcs[self.offsets[u]..self.offsets[u + 1]]
    }
    fn step(&self, p: &[f64], tau: f64) -> Vec<f64> {
        if self.n == 0 {
            return Vec::new();
        }
        let dangling: f64 = p
            .iter()
            .enumerate()
            .filter(|(u, _)| self.row(*u).is_empty())
            .map(|(_, v)| *v)
            .sum();
        let mut next = vec![(tau + (1. - tau) * dangling) / self.n as f64; self.n];
        for (u, &mass) in p.iter().enumerate() {
            for &(v, probability) in self.row(u) {
                next[v] += (1. - tau) * mass * probability;
            }
        }
        next
    }
    pub fn solve(&self, tau: f64, budget: usize, ctx: &Context) -> Result<Flow> {
        if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            return Err(anyhow!("invalid teleportation"));
        }
        if self.n == 0 {
            return Ok(Flow {
                visits: vec![],
                jumps: vec![],
                outgoing: vec![],
                incoming: vec![],
                iterations: 0,
                residual: 0.,
            });
        }
        let mut p = vec![1. / self.n as f64; self.n];
        let mut completed = None;
        // Lazy iteration removes periodic oscillation without changing stationary
        // flow. At tau=0 the uniform start selects a reducible-chain mixture.
        for iteration in 0..budget {
            if ctx.is_cancelled() {
                return Err(anyhow!("Infomap cancelled"));
            }
            let next = self.step(&p, tau);
            let residual: f64 = p.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
            if residual <= 1e-12 {
                completed = Some((iteration, residual));
                break;
            }
            for (mass, target) in p.iter_mut().zip(next) {
                *mass = 0.5 * (*mass + target);
            }
            let total: f64 = p.iter().sum();
            if !total.is_finite() || total <= 0. {
                return Err(anyhow!("invalid Infomap stationary mass"));
            }
            for mass in &mut p {
                *mass /= total;
            }
        }
        let (iterations, residual) = completed
            .ok_or_else(|| anyhow!("Infomap stationary flow did not converge within budget"))?;
        let mut outgoing = vec![Vec::new(); self.n];
        let mut incoming = vec![Vec::new(); self.n];
        let mut jumps = vec![0.; self.n];
        for u in 0..self.n {
            jumps[u] = p[u] * if self.row(u).is_empty() { 1. } else { tau };
            for &(v, probability) in self.row(u) {
                let mass = p[u] * (1. - tau) * probability;
                outgoing[u].push((v, mass));
                incoming[v].push((u, mass));
            }
        }
        Ok(Flow {
            visits: p,
            jumps,
            outgoing,
            incoming,
            iterations,
            residual,
        })
    }
}

pub(super) struct State {
    pub labels: Vec<usize>,
    sizes: Vec<usize>,
    masses: Vec<f64>,
    jumps: Vec<f64>,
    links: Vec<f64>,
    exits: Vec<f64>,
    exit_sum: f64,
    terms: f64,
    node_term: f64,
}
struct Candidate {
    label: usize,
    links: f64,
    exit: f64,
    mass: f64,
    jump: f64,
    size: usize,
}

impl Flow {
    pub fn state(&self, labels: Vec<usize>) -> Result<State> {
        let n = self.visits.len();
        if labels.len() != n || labels.iter().any(|&c| c >= n) {
            return Err(anyhow!("invalid map-equation partition"));
        }
        let mut sizes = vec![0; n];
        let mut masses = vec![0.; n];
        let mut jumps = vec![0.; n];
        let mut links = vec![0.; n];
        for (u, &c) in labels.iter().enumerate() {
            sizes[c] += 1;
            masses[c] += self.visits[u];
            jumps[c] += self.jumps[u];
            for &(v, f) in &self.outgoing[u] {
                if labels[v] != c {
                    links[c] += f;
                }
            }
        }
        let exits: Vec<_> = (0..n)
            .map(|c| links[c] + jumps[c] * (1. - sizes[c] as f64 / n.max(1) as f64))
            .collect();
        let exit_sum = exits.iter().sum();
        let terms = (0..n).map(|c| term(exits[c], masses[c])).sum();
        let node_term = self.visits.iter().map(|&v| xlog(v)).sum();
        let state = State {
            labels,
            sizes,
            masses,
            jumps,
            links,
            exits,
            exit_sum,
            terms,
            node_term,
        };
        state.score()?;
        Ok(state)
    }
    fn movement(&self, state: &State, u: usize) -> Result<Moves> {
        let old = state.labels[u];
        let mut out: BTreeMap<usize, f64> = BTreeMap::new();
        let mut incoming: BTreeMap<usize, f64> = BTreeMap::new();
        let mut total_out = 0.;
        for &(v, f) in &self.outgoing[u] {
            if v != u {
                *out.entry(state.labels[v]).or_default() += f;
                total_out += f;
            }
        }
        for &(v, f) in &self.incoming[u] {
            if v != u {
                *incoming.entry(state.labels[v]).or_default() += f;
            }
        }
        let delta = -total_out
            + out.get(&old).copied().unwrap_or(0.)
            + incoming.get(&old).copied().unwrap_or(0.);
        let source = state.candidate(
            old,
            state.sizes[old] - 1,
            state.masses[old] - self.visits[u],
            state.jumps[old] - self.jumps[u],
            state.links[old] + delta,
        )?;
        Ok(Moves {
            source,
            total_out,
            out,
            incoming,
        })
    }
    pub fn optimize(
        &self,
        trials: usize,
        max_iter: usize,
        rng: &mut fastrand::Rng,
        ctx: &mut Context,
    ) -> Result<Vec<usize>> {
        let n = self.visits.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // Always retain the one-codebook solution as a reference candidate.
        let mut best = self.state(vec![0; n])?;
        let mut best_score = best.score()?;
        ctx.record_stat("infomap.one_module_bits", best_score);
        for trial in 0..trials {
            let mut state = self.state((0..n).collect())?;
            let mut order: Vec<_> = (0..n).collect();
            for iteration in 0..max_iter {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Infomap cancelled"));
                }
                rng.shuffle(&mut order);
                let before = state.score()?;
                let mut moves = 0;
                for &u in &order {
                    if ctx.is_cancelled() {
                        return Err(anyhow!("Infomap cancelled"));
                    }
                    let old = state.labels[u];
                    let movement = self.movement(&state, u)?;
                    let mut chosen = None;
                    let mut chosen_score = state.score()?;
                    let free = state.sizes.iter().position(|&size| size == 0);
                    for target in 0..n {
                        if target == old || (state.sizes[target] == 0 && Some(target) != free) {
                            continue;
                        }
                        let dest = movement.target(self, &state, u, target)?;
                        let score = state.move_score(&movement.source, &dest)?;
                        if score < chosen_score - 1e-12 {
                            chosen = Some(dest);
                            chosen_score = score;
                        }
                    }
                    if let Some(dest) = chosen {
                        state.apply(u, movement.source, dest);
                        moves += 1;
                    }
                }
                // Rebuild from actual flows each sweep to bound incremental drift.
                state = self.state(state.labels)?;
                let after = state.score()?;
                if after > before + 1e-11 {
                    return Err(anyhow!("Infomap sweep increased map-equation length"));
                }
                ctx.record_stat(
                    format!("infomap.trial_{trial}.iteration_{iteration}.bits"),
                    after,
                );
                if moves == 0 {
                    break;
                }
            }
            let score = state.score()?;
            ctx.record_stat(format!("infomap.trial_{trial}.bits"), score);
            if score < best_score - 1e-12 {
                best = state;
                best_score = score;
            }
        }
        ctx.record_stat("infomap.best_bits", best_score);
        ctx.record_stat("infomap.trials", trials as f64);
        let mut remap = vec![usize::MAX; n];
        let mut next = 0;
        for c in &mut best.labels {
            if remap[*c] == usize::MAX {
                remap[*c] = next;
                next += 1;
            }
            *c = remap[*c];
        }
        Ok(best.labels)
    }
}

struct Moves {
    source: Candidate,
    total_out: f64,
    out: BTreeMap<usize, f64>,
    incoming: BTreeMap<usize, f64>,
}
impl Moves {
    fn target(&self, flow: &Flow, state: &State, u: usize, target: usize) -> Result<Candidate> {
        let delta = self.total_out
            - self.out.get(&target).copied().unwrap_or(0.)
            - self.incoming.get(&target).copied().unwrap_or(0.);
        state.candidate(
            target,
            state.sizes[target] + 1,
            state.masses[target] + flow.visits[u],
            state.jumps[target] + flow.jumps[u],
            state.links[target] + delta,
        )
    }
}

impl State {
    pub fn score(&self) -> Result<f64> {
        let score = xlog(self.exit_sum) + self.terms - self.node_term;
        if !score.is_finite() || score < -1e-10 {
            return Err(anyhow!("invalid Infomap code length"));
        }
        Ok(score.max(0.))
    }
    fn candidate(
        &self,
        label: usize,
        size: usize,
        mass: f64,
        jump: f64,
        links: f64,
    ) -> Result<Candidate> {
        let (mass, jump, links) = (nonnegative(mass)?, nonnegative(jump)?, nonnegative(links)?);
        let exit = links + jump * (1. - size as f64 / self.labels.len() as f64);
        Ok(Candidate {
            label,
            links,
            exit,
            mass,
            jump,
            size,
        })
    }
    fn move_score(&self, a: &Candidate, b: &Candidate) -> Result<f64> {
        let exits = nonnegative(
            self.exit_sum - self.exits[a.label] - self.exits[b.label] + a.exit + b.exit,
        )?;
        let terms = self.terms
            - term(self.exits[a.label], self.masses[a.label])
            - term(self.exits[b.label], self.masses[b.label])
            + term(a.exit, a.mass)
            + term(b.exit, b.mass);
        let score = xlog(exits) + terms - self.node_term;
        if !score.is_finite() || score < -1e-10 {
            return Err(anyhow!("invalid Infomap move length"));
        }
        Ok(score.max(0.))
    }
    fn apply(&mut self, u: usize, a: Candidate, b: Candidate) {
        let destination = b.label;
        for update in [a, b] {
            let c = update.label;
            self.exit_sum += update.exit - self.exits[c];
            self.terms += term(update.exit, update.mass) - term(self.exits[c], self.masses[c]);
            self.sizes[c] = update.size;
            self.masses[c] = update.mass;
            self.jumps[c] = update.jump;
            self.links[c] = update.links;
            self.exits[c] = update.exit;
        }
        self.labels[u] = destination;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entropy(rates: &[f64]) -> f64 {
        let total: f64 = rates.iter().sum();
        if total == 0. {
            return 0.;
        }
        rates
            .iter()
            .filter(|&&x| x > 0.)
            .map(|&x| {
                let p = x / total;
                -p * p.log2()
            })
            .sum()
    }
    // Direct dense transition matrix and separate index/module codebooks.
    fn reference(
        n: usize,
        edges: &[(usize, usize)],
        directed: bool,
        tau: f64,
        pi: &[f64],
        labels: &[usize],
    ) -> f64 {
        let mut a = vec![vec![0.; n]; n];
        for &(u, v) in edges {
            a[u][v] += 1.;
            if !directed {
                a[v][u] += 1.;
            }
        }
        let mut exits = vec![0.; n];
        let mut nodes: Vec<Vec<f64>> = vec![vec![]; n];
        for u in 0..n {
            nodes[labels[u]].push(pi[u]);
            let degree: f64 = a[u].iter().sum();
            for v in 0..n {
                let transition = if degree == 0. {
                    1. / n as f64
                } else {
                    tau / n as f64 + (1. - tau) * a[u][v] / degree
                };
                if labels[u] != labels[v] {
                    exits[labels[u]] += pi[u] * transition;
                }
            }
        }
        let mut result = exits.iter().sum::<f64>() * entropy(&exits);
        for c in 0..n {
            nodes[c].push(exits[c]);
            result += nodes[c].iter().sum::<f64>() * entropy(&nodes[c]);
        }
        result
    }
    #[test]
    fn objective_and_all_incremental_moves_match_direct_entropy_codebooks() {
        let edges = [(0, 0), (0, 1), (0, 1), (1, 2), (2, 0), (2, 3), (3, 2)];
        for directed in [false, true] {
            for tau in [0., 0.15, 1.] {
                let flow = Walk::new(4, edges, directed)
                    .unwrap()
                    .solve(tau, 10_000, &Context::new())
                    .unwrap();
                for code in 0_usize..81 {
                    let mut code = code;
                    let labels: Vec<_> = (0..4)
                        .map(|_| {
                            let c = code % 3;
                            code /= 3;
                            c
                        })
                        .collect();
                    let score = flow.state(labels.clone()).unwrap().score().unwrap();
                    assert!(
                        (score - reference(4, &edges, directed, tau, &flow.visits, &labels)).abs()
                            < 1e-11
                    );
                    for u in 0..4 {
                        for target in 0..4 {
                            if target == labels[u] {
                                continue;
                            }
                            let mut state = flow.state(labels.clone()).unwrap();
                            let movement = flow.movement(&state, u).unwrap();
                            let dest = movement.target(&flow, &state, u, target).unwrap();
                            let proposed = state.move_score(&movement.source, &dest).unwrap();
                            let mut moved = labels.clone();
                            moved[u] = target;
                            let expected =
                                reference(4, &edges, directed, tau, &flow.visits, &moved);
                            assert!((proposed - expected).abs() < 1e-11);
                            state.apply(u, movement.source, dest);
                            assert_eq!(state.labels, moved);
                            assert!((state.score().unwrap() - expected).abs() < 1e-11);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn stationary_flows_match_analytic_dangling_parallel_loop_and_periodic_cases() {
        let ctx = Context::new();
        let directed = Walk::new(2, [(0, 1)], true)
            .unwrap()
            .solve(0.25, 10_000, &ctx)
            .unwrap();
        assert!((directed.visits[0] - 4. / 11.).abs() < 1e-11);
        assert!((directed.visits[1] - 7. / 11.).abs() < 1e-11);
        let looped = Walk::new(2, [(0, 0), (0, 1), (0, 1)], false)
            .unwrap()
            .solve(0., 10_000, &ctx)
            .unwrap();
        assert!((looped.visits[0] - 2. / 3.).abs() < 1e-11);
        let periodic = Walk::new(3, [(0, 1), (1, 2)], false)
            .unwrap()
            .solve(0., 10_000, &ctx)
            .unwrap();
        for (got, want) in periodic.visits.iter().zip([0.25, 0.5, 0.25]) {
            assert!((got - want).abs() < 1e-11);
        }
        let absorbed = Walk::new(2, [(0, 1), (1, 1)], true)
            .unwrap()
            .solve(0., 10_000, &ctx)
            .unwrap();
        assert!(absorbed.visits[0] < 1e-11);
        let reducible = Walk::new(3, [(0, 0), (1, 1), (2, 0)], true)
            .unwrap()
            .solve(0., 10_000, &ctx)
            .unwrap();
        assert!((reducible.visits[0] - 2. / 3.).abs() < 1e-11);
        assert!((reducible.visits[1] - 1. / 3.).abs() < 1e-11);
        assert!(Walk::new(3, [(0, 1), (1, 2)], true)
            .unwrap()
            .solve(0.15, 0, &ctx)
            .is_err());
    }
    #[test]
    fn known_code_lengths_and_trial_prefixes_are_correct() {
        let ctx = Context::new();
        let flow = Walk::new(4, [(0, 1), (2, 3)], false)
            .unwrap()
            .solve(0., 10_000, &ctx)
            .unwrap();
        for (labels, want) in [
            (vec![0, 0, 1, 1], 1.),
            (vec![0, 0, 0, 0], 2.),
            (vec![0, 1, 2, 3], 4.),
        ] {
            assert!((flow.state(labels).unwrap().score().unwrap() - want).abs() < 1e-12);
        }
        let mut first = Context::new();
        let mut more = Context::new();
        flow.optimize(1, 20, &mut fastrand::Rng::with_seed(42), &mut first)
            .unwrap();
        flow.optimize(5, 20, &mut fastrand::Rng::with_seed(42), &mut more)
            .unwrap();
        assert!(
            more.stat_snapshot()["infomap.best_bits"]
                <= first.stat_snapshot()["infomap.best_bits"] + 1e-12
        );
        assert_eq!(more.stat_snapshot()["infomap.trials"], 5.);
    }
}
