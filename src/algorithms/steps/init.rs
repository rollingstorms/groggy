//! Initialization step primitives for creating node and edge variables.

use std::collections::HashMap;

use anyhow::Result;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

/// Initialize a node variable with a constant value for all nodes.
pub struct InitNodesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitNodesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitNodesStep {
    fn id(&self) -> &'static str {
        "core.init_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a node variable with a constant value".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let nodes = scope.subgraph().ordered_nodes();
        let mut map = HashMap::with_capacity(nodes.len());
        for &node in nodes.iter() {
            map.insert(node, self.value.clone());
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Initialize a node variable with sequential index values (0, 1, 2, ...).
/// Uses ordered_nodes() for deterministic ordering.
pub struct InitNodesWithIndexStep {
    target: String,
}

impl InitNodesWithIndexStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for InitNodesWithIndexStep {
    fn id(&self) -> &'static str {
        "core.init_nodes_with_index"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a node variable with sequential indices".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let nodes = scope.subgraph().ordered_nodes();
        let mut map = HashMap::with_capacity(nodes.len());

        for (idx, &node) in nodes.iter().enumerate() {
            map.insert(node, AlgorithmParamValue::Int(idx as i64));
        }

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Initialize a scalar variable with a constant value.
pub struct InitScalarStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitScalarStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitScalarStep {
    fn id(&self) -> &'static str {
        "core.init_scalar"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a scalar variable with a constant value".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), self.value.clone());
        Ok(())
    }
}

/// Initialize an edge variable with a constant value for all edges.
pub struct InitEdgesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitEdgesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitEdgesStep {
    fn id(&self) -> &'static str {
        "core.init_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize an edge variable with a constant".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for &edge in scope.edge_ids() {
            map.insert(edge, self.value.clone());
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

/// Get the number of nodes in the current subgraph as a scalar variable.
pub struct GraphNodeCountStep {
    target: String,
}

impl GraphNodeCountStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for GraphNodeCountStep {
    fn id(&self) -> &'static str {
        "core.graph_node_count"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Get the number of nodes in the graph".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let count = scope.subgraph().node_count();
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Int(count as i64));
        Ok(())
    }
}

/// Get the number of edges in the current subgraph as a scalar variable.
pub struct GraphEdgeCountStep {
    target: String,
}

impl GraphEdgeCountStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for GraphEdgeCountStep {
    fn id(&self) -> &'static str {
        "core.graph_edge_count"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Get the number of edges in the graph".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let count = scope.subgraph().edge_count();
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Int(count as i64));
        Ok(())
    }
}
