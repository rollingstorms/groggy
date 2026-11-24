//! Checked topology effects, published through one canonical mixed batch.
use super::core::{
    AtomicStepEffect, Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue,
};
use super::read_scope::{sealed, ReadStep, ReadStepScope};
use crate::algorithms::{AlgorithmParamValue, Context, CostHint};
use anyhow::{anyhow, bail, Result};

struct RemoveStep {
    nodes: bool,
    input: RemovalInput,
    effective_at: Option<crate::TimestampMicros>,
}

enum RemovalInput {
    Ids(Vec<usize>),
    /// Every key in a typed node/edge map; values are deliberately not predicates.
    Keys(String),
}

struct CaptureStagedStep {
    output: String,
}

struct ReviseEndpointsStep {
    rows: Vec<(usize, crate::BatchNodeRef, crate::BatchNodeRef)>,
    effective_at: Option<crate::TimestampMicros>,
}

impl Step for ReviseEndpointsStep {
    fn id(&self) -> &'static str {
        "graph.set_edge_endpoints"
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::Topology
    }
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

fn effective_time(spec: &StepSpec) -> Result<Option<crate::TimestampMicros>> {
    match spec.params.get("effective_at") {
        None => Ok(None),
        Some(AlgorithmParamValue::Int(at)) => Ok(Some(*at)),
        _ => bail!("effective_at must be a signed integer Unix timestamp in microseconds"),
    }
}

fn endpoint_rows(
    spec: &StepSpec,
) -> Result<Vec<(usize, crate::BatchNodeRef, crate::BatchNodeRef)>> {
    let rows = match spec.params.get("revisions") {
        Some(AlgorithmParamValue::Json(serde_json::Value::Array(rows))) => rows,
        Some(AlgorithmParamValue::IntList(rows)) if rows.is_empty() => return Ok(Vec::new()),
        _ => bail!("revisions must be an array of [edge_id, source, target] rows"),
    };
    rows.iter()
        .map(|row| {
            let row = row.as_array().filter(|row| row.len() == 3).ok_or_else(|| {
                anyhow!(
                    "each endpoint revision must contain an edge ID and two endpoint references"
                )
            })?;
            let id = |value: &serde_json::Value| {
                value
                    .as_u64()
                    .and_then(|id| usize::try_from(id).ok())
                    .ok_or_else(|| {
                        anyhow!("endpoint revision IDs must be nonnegative integers fitting usize")
                    })
            };
            Ok((
                id(&row[0])?,
                super::additions::endpoint(&row[1])?,
                super::additions::endpoint(&row[2])?,
            ))
        })
        .collect()
}

impl Step for CaptureStagedStep {
    fn id(&self) -> &'static str {
        "graph.capture_staged"
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::StagedCapture
    }
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl Step for RemoveStep {
    fn id(&self) -> &'static str {
        if self.nodes {
            "graph.remove_nodes"
        } else {
            "graph.remove_edges"
        }
    }

    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::Topology
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

fn instantiate(spec: &StepSpec, nodes: bool) -> Result<Box<dyn Step>> {
    for (key, _) in spec.params.iter() {
        if !matches!(key.as_str(), "ids" | "source" | "effective_at") {
            bail!("{} does not support parameter '{key}'", spec.id);
        }
    }
    let input = match (spec.params.get("ids"), spec.params.get("source")) {
        (Some(_), None) => RemovalInput::Ids(super::typed_values::ids(spec, "ids")?),
        (None, Some(AlgorithmParamValue::Text(source))) if !source.trim().is_empty() => {
            RemovalInput::Keys(source.clone())
        }
        _ => bail!(
            "{} requires exactly one of ids or a nonempty source name",
            spec.id
        ),
    };
    let effective_at = effective_time(spec)?;
    Ok(Box::new(RemoveStep {
        nodes,
        input,
        effective_at,
    }))
}

pub(super) fn register(registry: &StepRegistry) -> Result<()> {
    registry.register(
        "graph.set_edge_endpoints",
        StepMetadata {
            id: "graph.set_edge_endpoints".into(),
            description: "Stage existing edge endpoint revisions in one checked publication".into(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            for (key, _) in spec.params.iter() {
                if !matches!(key.as_str(), "revisions" | "effective_at") {
                    bail!("graph.set_edge_endpoints does not support parameter '{key}'");
                }
            }
            Ok(Box::new(ReviseEndpointsStep {
                rows: endpoint_rows(spec)?,
                effective_at: effective_time(spec)?,
            }))
        },
    )?;
    registry.register(
        "graph.capture_staged",
        StepMetadata {
            id: "graph.capture_staged".into(),
            description: "Capture a detached immutable selected pending graph state".into(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            for (key, _) in spec.params.iter() {
                if key != "output" {
                    bail!("graph.capture_staged does not support parameter '{key}'");
                }
            }
            let output = spec.params.expect_text("output")?;
            if output.trim().is_empty() {
                bail!("graph.capture_staged requires a nonempty output");
            }
            Ok(Box::new(CaptureStagedStep {
                output: output.into(),
            }))
        },
    )?;
    for (id, nodes) in [("graph.remove_nodes", true), ("graph.remove_edges", false)] {
        registry.register(
            id,
            StepMetadata {
                id: id.into(),
                description: "Stage terminal removals in one atomic graph publication".into(),
                cost_hint: CostHint::Linear,
            },
            move |spec| instantiate(spec, nodes),
        )?;
    }
    super::additions::register(registry)?;
    super::captured_algorithms::register(registry)?;
    Ok(())
}

impl sealed::Sealed for CaptureStagedStep {}
impl ReadStep for CaptureStagedStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            bail!("staged graph capture cancelled");
        }
        let view = scope.capture(ctx)?;
        if ctx.is_cancelled() {
            bail!("staged graph capture cancelled");
        }
        scope
            .variables_mut()
            .set_value(&self.output, StepValue::GraphView(view));
        Ok(())
    }
}

impl sealed::Sealed for RemoveStep {}
impl ReadStep for RemoveStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if !ctx.persist_results() || ctx.property_batch.is_none() {
            bail!("{} requires a persistent local atomic builder", self.id());
        }
        if ctx.is_cancelled() {
            bail!("topology removal cancelled");
        }
        let ids = match &self.input {
            RemovalInput::Ids(ids) => ids.clone(),
            RemovalInput::Keys(source) => {
                let mut ids = match (self.nodes, scope.variables().value(source)) {
                    (true, Some(StepValue::NodeMap(map))) => {
                        map.keys().copied().collect::<Vec<_>>()
                    }
                    (true, Some(StepValue::NodeColumn(column))) => column.nodes().to_vec(),
                    (false, Some(StepValue::EdgeMap(map))) => map.keys().copied().collect(),
                    (_, None) => bail!("variable '{source}' not found"),
                    _ => bail!(
                        "{} requires a matching node/edge map or node column",
                        self.id()
                    ),
                };
                ids.sort_unstable();
                ids
            }
        };
        scope.remove(ctx, self.nodes, ids, self.effective_at)?;
        Ok(())
    }
}

impl sealed::Sealed for ReviseEndpointsStep {}
impl ReadStep for ReviseEndpointsStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if !ctx.persist_results() || ctx.property_batch.is_none() {
            bail!("{} requires a persistent local atomic builder", self.id());
        }
        scope.revise_endpoints(ctx, &self.rows, self.effective_at)?;
        Ok(())
    }
}
