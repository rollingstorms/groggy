//! Algorithms over explicit immutable variables, including provisional captures.
use super::core::{
    AtomicStepEffect, Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue,
};
use crate::{
    algorithms::{
        centrality, community, AlgorithmParamValue as V, AlgorithmRead, Context, CostHint,
    },
    storage::table::Table,
    AttrValue,
};
use anyhow::{anyhow, bail, Result};
use std::sync::Arc;

struct CapturedAlgorithm {
    spec: StepSpec,
    source: String,
    output: String,
    algorithm: String,
}
impl CapturedAlgorithm {
    fn integer(&self, key: &str, default: i64) -> Result<i64> {
        match self.spec.params.get(key) {
            None => Ok(default),
            Some(V::Int(v)) => Ok(*v),
            _ => bail!("{key} must be an integer"),
        }
    }
    fn size(&self, key: &str, default: usize) -> Result<usize> {
        usize::try_from(self.integer(key, i64::try_from(default)?)?)
            .map_err(|_| anyhow!("{key} must be nonnegative and fit usize"))
    }
    fn float(&self, key: &str, default: f64) -> Result<f64> {
        match self.spec.params.get(key) {
            None => Ok(default),
            Some(V::Float(v)) => Ok(*v),
            Some(V::Int(v)) => Ok(*v as f64),
            _ => bail!("{key} must be numeric"),
        }
    }
    fn boolean(&self, key: &str, default: bool) -> Result<bool> {
        match self.spec.params.get(key) {
            None => Ok(default),
            Some(V::Bool(v)) => Ok(*v),
            _ => bail!("{key} must be boolean"),
        }
    }
    fn text(&self, key: &str) -> Result<Option<String>> {
        match self.spec.params.get(key) {
            None | Some(V::None) => Ok(None),
            Some(V::Text(v)) => Ok(Some(v.clone())),
            _ => bail!("{key} must be text or null"),
        }
    }
    fn seed(&self) -> Result<Option<u64>> {
        match self.spec.params.get("seed") {
            None | Some(V::None) => Ok(None),
            Some(V::Int(v)) => Ok(Some(u64::try_from(*v)?)),
            _ => bail!("seed must be a nonnegative integer or null"),
        }
    }
}
impl Step for CapturedAlgorithm {
    fn id(&self) -> &'static str {
        "graph.algorithm_table"
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }
    fn supports_atomic_properties(&self) -> bool {
        true
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            bail!("captured algorithm cancelled");
        }
        let view = scope.variables().graph_view(&self.source)?.clone();
        let read = AlgorithmRead::new(view.clone())?;
        let token = ctx.cancel_token();
        let cancel = token.as_deref();
        let label = self.output.clone();
        let weight = self.text("weight_attr")?;
        let mut diagnostics = Vec::new();
        let rows: Vec<(usize, AttrValue)> = match self.algorithm.as_str() {
            "bfs" => read
                .bfs_distances(self.size("start", 0)?, cancel)?
                .into_iter()
                .map(|(n, v)| (n, AttrValue::Int(v)))
                .collect(),
            "dfs" => read
                .dfs_order(self.size("start", 0)?, cancel)?
                .into_iter()
                .map(|(n, v)| (n, AttrValue::Int(v)))
                .collect(),
            "dijkstra" => read
                .dijkstra_distances(self.size("start", 0)?, weight.as_deref(), cancel)?
                .into_iter()
                .map(|(n, v)| (n, AttrValue::Float64(v)))
                .collect(),
            "astar" => {
                let heuristic = self.text("heuristic_attr")?;
                let path = read.astar_path(
                    self.size("start", 0)?,
                    self.size("goal", 0)?,
                    weight.as_deref(),
                    heuristic.as_deref(),
                    cancel,
                )?;
                diagnostics.push(("cost", AttrValue::Float64(path.total_weight)));
                path.nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, n)| Ok((n, AttrValue::Int(i64::try_from(i)?))))
                    .collect::<Result<_>>()?
            }
            "pagerank" => {
                let algorithm = centrality::PageRank::new(
                    self.float("damping", 0.85)?,
                    self.size("max_iter", 50)?,
                    self.float("tolerance", 1e-6)?,
                    self.text("personalization_attr")?,
                    label,
                )?;
                let result = read.pagerank(&algorithm, cancel)?;
                diagnostics.extend([
                    ("converged", AttrValue::Bool(result.converged)),
                    (
                        "iterations",
                        AttrValue::Int(i64::try_from(result.iterations)?),
                    ),
                    ("max_update", AttrValue::Float64(result.max_update)),
                ]);
                result
                    .scores
                    .into_iter()
                    .map(|(n, v)| (n, AttrValue::Float64(v)))
                    .collect()
            }
            "betweenness" => read
                .betweenness(
                    &centrality::BetweennessCentrality::new(
                        self.boolean("normalized", true)?,
                        weight,
                        label,
                    ),
                    cancel,
                )?
                .into_iter()
                .map(|(n, v)| (n, AttrValue::Float64(v)))
                .collect(),
            "closeness" => {
                let algorithm = centrality::ClosenessCentrality::new(
                    self.boolean("harmonic", true)?,
                    weight,
                    label,
                )
                .with_normalization(self.boolean("normalized", false)?)
                .with_direction(
                    match self.text("direction")?.as_deref().unwrap_or("outgoing") {
                        "outgoing" => centrality::ClosenessDirection::Outgoing,
                        "incoming" => centrality::ClosenessDirection::Incoming,
                        _ => bail!("direction must be outgoing or incoming"),
                    },
                );
                read.closeness(&algorithm, cancel)?
                    .into_iter()
                    .map(|(n, v)| (n, AttrValue::Float64(v)))
                    .collect()
            }
            "components" => {
                let mode = match self.text("mode")?.as_deref().unwrap_or("undirected") {
                    "undirected" => community::ComponentMode::Undirected,
                    "weak" => community::ComponentMode::Weak,
                    "strong" => community::ComponentMode::Strong,
                    _ => bail!("mode must be undirected, weak or strong"),
                };
                let mut rows = Vec::new();
                for (label, component) in read
                    .connected_components(mode, cancel)?
                    .into_iter()
                    .enumerate()
                {
                    for node in component.node_ids() {
                        rows.push((node, AttrValue::Int(i64::try_from(label)?)));
                    }
                }
                rows.sort_unstable_by_key(|(node, _)| *node);
                rows
            }
            "louvain" | "leiden" | "infomap" | "girvan_newman" => {
                let rows = match self.algorithm.as_str() {
                    "louvain" => read.louvain(
                        &community::Louvain::new(
                            self.size("max_iter", 20)?,
                            self.size("max_phases", 10)?,
                            self.float("resolution", 1.0)?,
                            label,
                        )?,
                        ctx,
                    )?,
                    "leiden" => read.leiden(
                        &community::Leiden::new(
                            self.size("max_iter", 20)?,
                            self.size("max_phases", 10)?,
                            self.float("resolution", 1.0)?,
                            self.seed()?,
                            label,
                        )?
                        .with_theta(self.float("theta", 0.01)?)?,
                        ctx,
                    )?,
                    "infomap" => read.infomap(
                        &community::Infomap::new(
                            self.float("teleportation", 0.15)?,
                            self.size("num_trials", 10)?,
                            self.size("max_iter", 100)?,
                            self.seed()?,
                            label,
                        )?
                        .with_flow_budget(self.size("flow_max_iter", 10000)?)?,
                        ctx,
                    )?,
                    _ => read.girvan_newman(
                        &community::GirvanNewman::new(
                            if self.spec.params.contains_key("num_levels") {
                                Some(self.size("num_levels", 0)?)
                            } else {
                                None
                            },
                            if self.spec.params.contains_key("modularity_threshold") {
                                Some(self.float("modularity_threshold", 0.0001)?)
                            } else {
                                None
                            },
                            weight,
                            label,
                        ),
                        ctx,
                    )?,
                };
                rows.into_iter()
                    .map(|(n, v)| Ok((n, AttrValue::Int(i64::try_from(v)?))))
                    .collect::<Result<_>>()?
            }
            "lpa" => read.label_propagation(
                &community::LabelPropagation::new(
                    self.size("max_iter", 20)?,
                    self.float("tolerance", 0.01)?,
                    label,
                    self.text("seed_attr")?,
                )?
                .with_seed(self.seed()?),
                ctx,
            )?,
            _ => bail!("unsupported captured algorithm"),
        };
        if ctx.is_cancelled() {
            bail!("captured algorithm cancelled");
        }
        let nodes = rows.iter().map(|(node, _)| *node).collect::<Vec<_>>();
        let mut table =
            crate::storage::table::projection::nodes(&nodes, Vec::new(), |_, _| Ok(None))?
                .with_graph_source(view, true, nodes)?
                .into_base_table()?;
        table.set_column("value".into(), rows.into_iter().map(|(_, v)| v).collect())?;
        for (name, value) in diagnostics {
            table.set_column(name.into(), vec![value; table.nrows()])?;
        }
        scope
            .variables_mut()
            .set_value(&self.output, StepValue::Table(Arc::new(table)));
        Ok(())
    }
}
pub(super) fn register(registry: &StepRegistry) -> Result<()> {
    registry.register(
        "graph.algorithm_table",
        StepMetadata {
            id: "graph.algorithm_table".into(),
            description: "Compute through canonical kernels on an explicit immutable graph input"
                .into(),
            cost_hint: CostHint::Unknown,
        },
        |spec| {
            let algorithm = spec.params.expect_text("algorithm")?.to_owned();
            let options: &[&str] = match algorithm.as_str() {
                "bfs" | "dfs" => &["start"],
                "dijkstra" => &["start", "weight_attr"],
                "astar" => &["start", "goal", "weight_attr", "heuristic_attr"],
                "pagerank" => &["damping", "max_iter", "tolerance", "personalization_attr"],
                "betweenness" => &["normalized", "weight_attr"],
                "closeness" => &["normalized", "harmonic", "weight_attr", "direction"],
                "components" => &["mode"],
                "louvain" => &["max_iter", "max_phases", "resolution"],
                "leiden" => &["max_iter", "max_phases", "resolution", "seed", "theta"],
                "infomap" => &[
                    "max_iter",
                    "num_trials",
                    "teleportation",
                    "seed",
                    "flow_max_iter",
                ],
                "girvan_newman" => &["num_levels", "modularity_threshold", "weight_attr"],
                "lpa" => &["max_iter", "tolerance", "seed_attr", "seed"],
                _ => bail!("unsupported captured algorithm '{algorithm}'"),
            };
            for key in spec.params.keys() {
                if !matches!(key.as_str(), "algorithm" | "source" | "output")
                    && !options.contains(&key.as_str())
                {
                    bail!("{algorithm} does not support parameter '{key}'");
                }
            }
            for key in ["start", "goal"]
                .iter()
                .filter(|&&key| options.contains(&key))
            {
                if !spec.params.contains_key(*key) {
                    bail!("{algorithm} requires {key}");
                }
            }
            let source = spec.params.expect_text("source")?.to_owned();
            let output = spec.params.expect_text("output")?.to_owned();
            if source.trim().is_empty() || output.trim().is_empty() {
                bail!("source and output must be nonempty");
            }
            Ok(Box::new(CapturedAlgorithm {
                spec: spec.clone(),
                source,
                output,
                algorithm,
            }))
        },
    )?;
    Ok(())
}
