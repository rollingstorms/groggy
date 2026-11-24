//! Typed immutable graph/table stages over explicit captured inputs.
//! No working-graph clone, implicit publication or staged-property recapture.
use super::core::{
    AtomicStepEffect, Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue,
};
use super::read_scope::{sealed, ReadStep, ReadStepScope};
use crate::{
    algorithms::{AlgorithmOutput, AlgorithmParamValue, Context, CostHint},
    storage::table::Table,
    GraphView,
};
use anyhow::{anyhow, bail, Result};
use std::sync::Arc;

enum Operation {
    FromSnapshot,
    SelectNodes(Vec<usize>),
    SelectEdges(Vec<usize>),
    NodesTable,
    EdgesTable,
    DegreesTable,
    SelectColumns(Vec<String>),
    SelectRows(Vec<usize>),
    EmitGraph,
    EmitTable,
}

struct TypedReadStep {
    id: &'static str,
    source: String,
    output: String,
    operation: Operation,
}

fn check_cancelled(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        bail!("typed graph/table stage cancelled");
    }
    Ok(())
}

impl Step for TypedReadStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }
    fn supports_atomic_properties(&self) -> bool {
        // Explicit immutable inputs are independent of in-flight property writes.
        true
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

fn degrees_table(ctx: &Context, view: &GraphView) -> Result<crate::storage::table::BaseTable> {
    use crate::AttrValue;
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;
    let nodes = view.node_ids();
    let token = ctx.cancel_token();
    let counts = |node| -> Result<[AttrValue; 3]> {
        if token
            .as_ref()
            .is_some_and(|token| token.load(Ordering::Relaxed))
        {
            bail!("degree table cancelled");
        }
        let count = |value: usize| -> Result<AttrValue> {
            Ok(AttrValue::Int(i64::try_from(value).map_err(|_| {
                anyhow!("degree exceeds table integer range")
            })?))
        };
        // Reuse the captured view's selected edge-instance conventions. There
        // is no lookup against the writable owner or dense positional node IDs.
        Ok([
            count(view.degree(node)?)?,
            count(view.in_degree(node)?)?,
            count(view.out_degree(node)?)?,
        ])
    };
    let mut incident = Vec::with_capacity(nodes.len());
    let mut incoming = Vec::with_capacity(nodes.len());
    let mut outgoing = Vec::with_capacity(nodes.len());
    if nodes.len() >= 4096 {
        incident.resize(nodes.len(), AttrValue::Null);
        incoming.resize(nodes.len(), AttrValue::Null);
        outgoing.resize(nodes.len(), AttrValue::Null);
        // Only detached captured inputs, cancellation and disjoint owned cells
        // cross workers. Context and the writable owner stay on their thread.
        nodes
            .par_iter()
            .zip(incident.par_iter_mut())
            .zip(incoming.par_iter_mut())
            .zip(outgoing.par_iter_mut())
            .try_for_each(|(((&node, incident), incoming), outgoing)| -> Result<()> {
                let [degree, indegree, outdegree] = counts(node)?;
                *incident = degree;
                *incoming = indegree;
                *outgoing = outdegree;
                Ok(())
            })?;
    } else {
        for &node in &nodes {
            let [degree, indegree, outdegree] = counts(node)?;
            incident.push(degree);
            incoming.push(indegree);
            outgoing.push(outdegree);
        }
    }
    let mut table = crate::storage::table::projection::nodes(&nodes, Vec::new(), |_, _| Ok(None))?
        .with_graph_source(view.clone(), true, nodes)?
        .into_base_table()?;
    for (name, values) in [
        ("degree", incident),
        ("in_degree", incoming),
        ("out_degree", outgoing),
    ] {
        table.set_column(name, values)?;
    }
    Ok(table)
}

fn text(spec: &StepSpec, key: &str) -> Result<String> {
    let value = spec.params.expect_text(key)?;
    if value.trim().is_empty() {
        bail!("{} requires a nonempty '{key}'", spec.id);
    }
    Ok(value.to_owned())
}

fn array<'a>(spec: &'a StepSpec, key: &str) -> Result<&'a [serde_json::Value]> {
    match spec.params.get(key) {
        Some(AlgorithmParamValue::Json(serde_json::Value::Array(values))) => Ok(values),
        _ => Err(anyhow!("{} requires a JSON array for '{key}'", spec.id)),
    }
}

pub(super) fn ids(spec: &StepSpec, key: &str) -> Result<Vec<usize>> {
    // Builder lowering preserves homogeneous integer arrays as IntList.
    // Direct registry callers may supply the original JSON array instead.
    if let Some(AlgorithmParamValue::IntList(values)) = spec.params.get(key) {
        return values
            .iter()
            .map(|&id| {
                usize::try_from(id).map_err(|_| {
                    anyhow!(
                        "{} '{key}' requires nonnegative integer IDs/positions fitting usize",
                        spec.id
                    )
                })
            })
            .collect();
    }
    array(spec, key)?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| {
                    anyhow!(
                        "{} '{key}' requires nonnegative integer IDs/positions fitting usize",
                        spec.id
                    )
                })
        })
        .collect()
}

fn columns(spec: &StepSpec) -> Result<Vec<String>> {
    if let Some(AlgorithmParamValue::TextList(values)) = spec.params.get("columns") {
        return Ok(values.clone());
    }
    array(spec, "columns")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("{} columns must be strings", spec.id))
        })
        .collect()
}

pub(super) fn register(registry: &StepRegistry) -> Result<()> {
    for &id in &[
        "graph.from_snapshot",
        "graph.select_nodes",
        "graph.select_edges",
        "graph.nodes_table",
        "graph.edges_table",
        "graph.degrees_table",
        "table.select_columns",
        "table.select_rows",
        "graph.emit",
        "table.emit",
    ] {
        registry.register(
            id,
            StepMetadata {
                id: id.into(),
                description: "Typed immutable graph/table stage".into(),
                cost_hint: CostHint::Linear,
            },
            move |spec| {
                let emit = matches!(id, "graph.emit" | "table.emit");
                let target = if emit { "name" } else { "output" };
                let extra = match id {
                    "graph.select_nodes" | "graph.select_edges" => Some("ids"),
                    "table.select_rows" => Some("rows"),
                    "table.select_columns" => Some("columns"),
                    _ => None,
                };
                for (key, _) in spec.params.iter() {
                    if key != "source" && key != target && Some(key.as_str()) != extra {
                        bail!("{id} does not support parameter '{key}'");
                    }
                }
                let operation = match id {
                    "graph.from_snapshot" => Operation::FromSnapshot,
                    "graph.select_nodes" => Operation::SelectNodes(ids(spec, "ids")?),
                    "graph.select_edges" => Operation::SelectEdges(ids(spec, "ids")?),
                    "graph.nodes_table" => Operation::NodesTable,
                    "graph.edges_table" => Operation::EdgesTable,
                    "graph.degrees_table" => Operation::DegreesTable,
                    "table.select_rows" => Operation::SelectRows(ids(spec, "rows")?),
                    "table.select_columns" => Operation::SelectColumns(columns(spec)?),
                    "graph.emit" => Operation::EmitGraph,
                    "table.emit" => Operation::EmitTable,
                    _ => bail!("unsupported typed stage '{id}'"),
                };
                Ok(Box::new(TypedReadStep {
                    id,
                    source: text(spec, "source")?,
                    output: text(spec, target)?,
                    operation,
                }))
            },
        )?;
    }
    Ok(())
}

impl sealed::Sealed for TypedReadStep {}
impl ReadStep for TypedReadStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        check_cancelled(ctx)?;
        let vars = scope.variables();
        let value = match &self.operation {
            Operation::FromSnapshot => StepValue::GraphView(GraphView::all(
                vars.snapshot(&self.source)?.read_state().clone(),
            )),
            Operation::SelectNodes(ids) => {
                StepValue::GraphView(vars.graph_view(&self.source)?.select_nodes(ids)?)
            }
            Operation::SelectEdges(ids) => {
                StepValue::GraphView(vars.graph_view(&self.source)?.select_edges(ids)?)
            }
            Operation::NodesTable => StepValue::Table(Arc::new(
                vars.graph_view(&self.source)?
                    .node_table_data()?
                    .into_base_table()?,
            )),
            Operation::EdgesTable => StepValue::Table(Arc::new(
                vars.graph_view(&self.source)?
                    .edge_table_data()?
                    .into_base_table()?,
            )),
            Operation::DegreesTable => StepValue::Table(Arc::new(degrees_table(
                ctx,
                vars.graph_view(&self.source)?,
            )?)),
            Operation::SelectColumns(names) => {
                StepValue::Table(Arc::new(vars.table(&self.source)?.select(names)?))
            }
            Operation::SelectRows(rows) => {
                StepValue::Table(Arc::new(vars.table(&self.source)?.select_rows(rows)?))
            }
            Operation::EmitGraph => {
                let view = vars.graph_view(&self.source)?.clone();
                check_cancelled(ctx)?;
                ctx.add_output(&self.output, AlgorithmOutput::GraphView(view));
                return Ok(());
            }
            Operation::EmitTable => {
                let table = match vars.value(&self.source) {
                    Some(StepValue::Table(table)) => Arc::clone(table),
                    Some(_) => bail!("variable '{}' is not a table", self.source),
                    None => bail!("variable '{}' not found", self.source),
                };
                check_cancelled(ctx)?;
                ctx.add_output(&self.output, AlgorithmOutput::Table(table));
                return Ok(());
            }
        };
        // Install only fully computed values. Failed selection/projection or
        // cancellation leaves a pre-existing target variable intact.
        check_cancelled(ctx)?;
        scope.variables_mut().set_value(&self.output, value);
        Ok(())
    }
}
