//! Typed immutable graph/table stages over explicit captured inputs.
//! No working-graph clone, implicit publication or staged-property recapture.
use super::core::{Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue};
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
    fn supports_atomic_properties(&self) -> bool {
        // Explicit immutable inputs are independent of in-flight property writes.
        true
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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
