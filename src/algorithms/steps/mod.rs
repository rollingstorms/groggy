//! Step primitives used by the pipeline interpreter.
//!
//! Steps are intentionally fine-grained so that Python builders and future
//! planners can assemble bespoke algorithms while still executing entirely in
//! Rust. Phase 1 provides a small core set focused on node transformations and
//! attribute attachment.

// Module declarations
mod aggregations;
mod arithmetic;
mod attributes;
mod community;
pub mod composition;
mod core;
pub mod direction;
mod execution_block;
mod expression;
mod filtering;
mod flow;
mod fused;
mod init;
mod loop_step;
mod normalization;
mod pathfinding;
mod registry;
mod sampling;
pub mod schema;
mod structural;
pub mod temporal;
mod transformations;
pub mod validation;

// Re-export core types
pub use core::{
    global_step_registry, Step, StepEdgeInput, StepInput, StepMetadata, StepRegistry, StepScope,
    StepSpec, StepValue, StepVariables,
};

// Re-export expression types
pub use expression::{BinaryOp, Expr, ExprContext, UnaryOp};

// Re-export filtering types
pub use filtering::{
    FilterEdgesByAttrStep, FilterNodesByAttrStep, Predicate, SortNodesByAttrStep, SortOrder,
    TopKStep,
};

// Re-export step implementations
pub use aggregations::{
    EntropyStep, HistogramStep, MedianStep, ModeStep, QuantileStep, ReduceNodeValuesStep,
    Reduction, StdDevStep,
};
pub use arithmetic::{
    BinaryArithmeticStep, BroadcastScalarStep, CollectNeighborValuesStep, CompareOp, CompareStep,
    ModeListStep, ModeTieBreak, RecipStep, ReduceScalarStep, ReductionOp, WhereStep,
};
pub use attributes::{
    AttachEdgeAttrStep, AttachNodeAttrStep, EdgeWeightScaleStep, LoadEdgeAttrStep, LoadNodeAttrStep,
};
pub use community::{CommunitySeedStep, LabelPropagateStep, ModularityGainStep, SeedStrategy};
pub use flow::{AliasStep, FlowUpdateStep, ResidualCapacityStep};
pub use init::{
    GraphEdgeCountStep, GraphNodeCountStep, InitEdgesStep, InitNodesStep, InitNodesWithIndexStep,
    InitScalarStep,
};
pub use normalization::{
    ClipValuesStep, NormalizeMethod, NormalizeNodeValuesStep, NormalizeValuesStep, StandardizeStep,
};
pub use pathfinding::{KShortestPathsStep, RandomWalkStep, ShortestPathMapStep};
pub use sampling::{EntityType, ReservoirSampleStep, SampleEdgesStep, SampleNodesStep, SampleSpec};
pub use structural::{
    EdgeWeightSumStep, KCoreMarkStep, NodeDegreeStep, TriangleCountStep, WeightedDegreeStep,
};
pub use transformations::{MapNodesExprStep, MapNodesStep, NeighborModeUpdateStep, NodeMapFn};

// Re-export fused operations
pub use fused::{FusedAXPY, FusedMADD, FusedNeighborMulAgg};

// Re-export direction types
pub use direction::NeighborDirection;

// Re-export execution block types
pub use execution_block::{BlockBody, BlockOptions, BodyNode, ExecutionBlockStep, ExecutionMode};

// Re-export temporal steps
pub use temporal::{
    AggregateFunction, DiffEdgesStep, DiffNodesStep, MarkChangedNodesStep, TemporalFilterStep,
    TemporalPredicate, WindowAggregateStep,
};

// Re-export registration functions
pub use registry::{ensure_core_steps_registered, register_core_steps};

// Re-export schema and validation types
pub use schema::{
    Constraint, ParameterSchema, ParameterSchemaBuilder, ParameterType, SchemaRegistry, StepSchema,
    StepSchemaBuilder,
};
pub use validation::{
    ErrorCategory, PipelineValidator, ValidationError, ValidationReport, ValidationWarning,
};

// Re-export composition helpers
pub use composition::{StepComposer, StepTemplate};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::steps::temporal::{SnapshotAtStep, TemporalWindowStep};
    use crate::algorithms::temporal::TemporalScope;
    use crate::algorithms::AlgorithmParamValue;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use crate::traits::SubgraphOperations;
    use crate::types::{AttrName, AttrValue, NodeId};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::rc::Rc;
    use std::sync::Arc;

    fn sample_subgraph() -> (Subgraph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let nodes = vec![a, b];
        let set: HashSet<NodeId> = nodes.iter().copied().collect();
        let sg =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), set, "test".to_string()).unwrap();
        (sg, nodes)
    }

    #[test]
    fn init_nodes_sets_values() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = InitNodesStep::new("labels", AlgorithmParamValue::Int(1));
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let map = scope.variables().node_map("labels").unwrap();
        for node in nodes {
            assert_eq!(map.get(&node), Some(&AlgorithmParamValue::Int(1)));
        }
    }

    #[test]
    fn load_node_attr_fetches_values() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        graph
            .set_node_attr(a, "score".into(), AttrValue::Float(0.5))
            .unwrap();
        graph
            .set_node_attr(b, "score".into(), AttrValue::Float(1.5))
            .unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = LoadNodeAttrStep::new(
            "scores",
            AttrName::from("score".to_string()),
            AlgorithmParamValue::Float(0.0),
        );
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("scores").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Float(0.5)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Float(1.5)));
    }

    #[test]
    fn load_entity_id_defaults_to_node_id() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = LoadNodeAttrStep::new(
            "entity_ids",
            AttrName::from("entity_id".to_string()),
            AlgorithmParamValue::Float(0.0),
        );
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("entity_ids").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(a as i64)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(b as i64)));
    }

    #[test]
    fn attach_node_attr_persists_values() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Float(0.5)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = AttachNodeAttrStep::new("scores", AttrName::from("score".to_string()));
        step.apply(&mut Context::new(), &mut scope).unwrap();
        for &node in &nodes {
            let val = sg
                .get_node_attribute(node, &AttrName::from("score".to_string()))
                .unwrap();
            assert!(val.is_some());
        }
    }

    #[test]
    fn binary_arithmetic_add_maps() {
        use crate::algorithms::steps::arithmetic::BinaryOperation;
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "left",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Int(2)))
                .collect(),
        );
        vars.set_node_map(
            "right",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Int(3)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step =
            BinaryArithmeticStep::new("core.add", "left", "right", "sum", BinaryOperation::Add);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("sum").unwrap();
        for node in nodes {
            assert_eq!(map.get(&node), Some(&AlgorithmParamValue::Int(5)));
        }
    }

    #[test]
    fn binary_arithmetic_add_strings() {
        use crate::algorithms::steps::arithmetic::BinaryOperation;
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "prefix",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Text("a".to_string())))
                .collect(),
        );
        vars.set_node_map(
            "suffix",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Text("b".to_string())))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = BinaryArithmeticStep::new(
            "core.add",
            "prefix",
            "suffix",
            "joined",
            BinaryOperation::Add,
        );
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("joined").unwrap();
        for node in nodes {
            assert_eq!(
                map.get(&node),
                Some(&AlgorithmParamValue::Text("ab".to_string()))
            );
        }
    }

    #[test]
    fn node_degree_step_computes_degrees() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = NodeDegreeStep::new("degree", None);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("degree").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(2)));
        assert_eq!(map.get(&c), Some(&AlgorithmParamValue::Int(1)));
    }

    #[test]
    fn normalize_node_values_step_scales_values() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AlgorithmParamValue::Int((idx + 1) as i64)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step =
            NormalizeNodeValuesStep::new("scores", "norm_scores", NormalizeMethod::Sum, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("norm_scores").unwrap();
        let total: f64 = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_node_values_step_max_method() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AlgorithmParamValue::Int((idx + 1) as i64)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = NormalizeNodeValuesStep::new("scores", "max_scores", NormalizeMethod::Max, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("max_scores").unwrap();
        let max = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_node_values_step_minmax_method() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AlgorithmParamValue::Int((idx + 1) as i64)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step =
            NormalizeNodeValuesStep::new("scores", "range_scores", NormalizeMethod::MinMax, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("range_scores").unwrap();
        let min = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::INFINITY, f64::min);
        let max = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ensure_core_steps_registers_once() {
        ensure_core_steps_registered();
        ensure_core_steps_registered();
        let registry = global_step_registry();
        assert!(registry.contains("core.init_nodes"));
        assert!(registry.contains("core.attach_edge_attr"));
        assert!(registry.contains("core.load_attr"));
        assert!(registry.contains("core.add"));
        assert!(registry.contains("core.weighted_degree"));
        assert!(registry.contains("core.k_core_mark"));
        assert!(registry.contains("core.triangle_count"));
    }

    #[test]
    fn weighted_degree_step_computes_weighted_degrees() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let e1 = graph.add_edge(a, b).unwrap();
        let e2 = graph.add_edge(b, c).unwrap();

        // Set edge weights
        graph
            .set_edge_attr(e1, "weight".into(), AttrValue::Float(2.0))
            .unwrap();
        graph
            .set_edge_attr(e2, "weight".into(), AttrValue::Float(3.0))
            .unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = WeightedDegreeStep::new(AttrName::from("weight".to_string()), "weighted_deg");
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("weighted_deg").unwrap();

        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Float(2.0)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Float(5.0))); // 2.0 + 3.0
        assert_eq!(map.get(&c), Some(&AlgorithmParamValue::Float(3.0)));
    }

    #[test]
    fn k_core_mark_step_identifies_cores() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();

        // Create a triangle (3-core) plus one peripheral node
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();
        graph.add_edge(d, a).unwrap(); // d only connected to a

        let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);

        // Test 2-core: all nodes should be included
        let step = KCoreMarkStep::new(2, "k2_core");
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("k2_core").unwrap();

        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&c), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&d), Some(&AlgorithmParamValue::Int(0))); // Not in 2-core
    }

    #[test]
    fn triangle_count_step_counts_triangles() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();

        // Create one triangle: a-b-c
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();
        // Add one edge outside triangle
        graph.add_edge(d, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = TriangleCountStep::new("triangles");
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("triangles").unwrap();

        // Each node in the triangle should have count 1
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&c), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&d), Some(&AlgorithmParamValue::Int(0)));
    }

    #[test]
    fn normalize_values_step_works_on_edges() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let e1 = graph.add_edge(a, b).unwrap();
        let e2 = graph.add_edge(b, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();

        // Create edge map with values 3.0 and 7.0 (sum = 10.0)
        let mut edge_map = HashMap::new();
        edge_map.insert(e1, AlgorithmParamValue::Float(3.0));
        edge_map.insert(e2, AlgorithmParamValue::Float(7.0));
        vars.set_edge_map("weights", edge_map);

        let mut scope = StepScope::new(&sg, &mut vars);
        let step = NormalizeValuesStep::new("weights", "normalized", NormalizeMethod::Sum, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().edge_map("normalized").unwrap();
        assert_eq!(result.get(&e1), Some(&AlgorithmParamValue::Float(0.3)));
        assert_eq!(result.get(&e2), Some(&AlgorithmParamValue::Float(0.7)));
    }

    #[test]
    fn standardize_step_computes_z_scores() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        // Create values with known mean and std
        // Values: 10, 20 -> mean = 15, std = 5
        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Float(10.0));
        map.insert(nodes[1], AlgorithmParamValue::Float(20.0));
        vars.set_node_map("scores", map);

        let mut scope = StepScope::new(&sg, &mut vars);
        let step = StandardizeStep::new("scores", "z_scores", 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("z_scores").unwrap();
        // Z-scores should be -1.0 and 1.0
        if let Some(AlgorithmParamValue::Float(z1)) = result.get(&nodes[0]) {
            assert!((z1 + 1.0).abs() < 1e-6);
        } else {
            panic!("Expected Float z-score");
        }
        if let Some(AlgorithmParamValue::Float(z2)) = result.get(&nodes[1]) {
            assert!((z2 - 1.0).abs() < 1e-6);
        } else {
            panic!("Expected Float z-score");
        }
    }

    #[test]
    fn clip_values_step_clamps_range() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Float(5.0));
        map.insert(nodes[1], AlgorithmParamValue::Float(15.0));
        vars.set_node_map("values", map);

        let mut scope = StepScope::new(&sg, &mut vars);
        let step = ClipValuesStep::new("values", "clipped", 8.0, 12.0);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("clipped").unwrap();
        assert_eq!(
            result.get(&nodes[0]),
            Some(&AlgorithmParamValue::Float(8.0))
        ); // Clipped to min
        assert_eq!(
            result.get(&nodes[1]),
            Some(&AlgorithmParamValue::Float(12.0))
        ); // Clipped to max
    }

    #[test]
    fn map_nodes_expr_doubles_values() {
        use crate::algorithms::steps::expression::{BinaryOp, Expr};

        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Float(5.0));
        map.insert(nodes[1], AlgorithmParamValue::Float(10.0));
        vars.set_node_map("values", map);

        let mut scope = StepScope::new(&sg, &mut vars);

        // Create expression: value * 2
        let expr = Expr::binary(
            BinaryOp::Mul,
            Expr::var("value"),
            Expr::constant(AlgorithmParamValue::Float(2.0)),
        );

        let step = MapNodesExprStep::new("values", "doubled", expr);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("doubled").unwrap();
        assert_eq!(
            result.get(&nodes[0]),
            Some(&AlgorithmParamValue::Float(10.0))
        );
        assert_eq!(
            result.get(&nodes[1]),
            Some(&AlgorithmParamValue::Float(20.0))
        );
    }

    #[test]
    fn map_nodes_expr_uses_neighbor_count() {
        use crate::algorithms::steps::expression::Expr;

        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Int(0));
        map.insert(nodes[1], AlgorithmParamValue::Int(0));
        vars.set_node_map("values", map);

        let mut scope = StepScope::new(&sg, &mut vars);

        // Create expression: neighbor_count()
        let expr = Expr::Call {
            func: "neighbor_count".to_string(),
            args: vec![],
        };

        let step = MapNodesExprStep::new("values", "degrees", expr);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("degrees").unwrap();
        // Nodes should have their degree as value
        assert!(matches!(
            result.get(&nodes[0]),
            Some(AlgorithmParamValue::Int(_))
        ));
        assert!(matches!(
            result.get(&nodes[1]),
            Some(AlgorithmParamValue::Int(_))
        ));
    }

    #[test]
    fn diff_nodes_step_detects_changes() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        let b = graph.add_node();
        let commit2 = graph.commit("c2".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let before = graph_rc.borrow().snapshot_at_commit(commit1).unwrap();
        let after = graph_rc.borrow().snapshot_at_commit(commit2).unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        vars.set_snapshot("before", before);
        vars.set_snapshot("after", after);
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx = Context::new();

        let step = DiffNodesStep::new(
            Some("before".to_string()),
            Some("after".to_string()),
            "delta".to_string(),
        );
        step.apply(&mut ctx, &mut scope).unwrap();

        let added = scope.variables().node_map("delta_nodes_added").unwrap();
        assert_eq!(added.get(&b), Some(&AlgorithmParamValue::Int(1)));
        let removed = scope.variables().node_map("delta_nodes_removed").unwrap();
        assert!(removed.is_empty());
    }

    #[test]
    fn diff_edges_step_detects_changes() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        graph.add_edge(a, b).unwrap();
        let commit2 = graph.commit("c2".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let before = graph_rc.borrow().snapshot_at_commit(commit1).unwrap();
        let after = graph_rc.borrow().snapshot_at_commit(commit2).unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        vars.set_snapshot("before", before);
        vars.set_snapshot("after", after);
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx = Context::new();

        let step = DiffEdgesStep::new(
            Some("before".to_string()),
            Some("after".to_string()),
            "delta".to_string(),
        );
        step.apply(&mut ctx, &mut scope).unwrap();

        let added = scope.variables().edge_map("delta_edges_added").unwrap();
        assert_eq!(added.len(), 1);
        let removed = scope.variables().edge_map("delta_edges_removed").unwrap();
        assert!(removed.is_empty());
    }

    #[test]
    fn window_aggregate_sums_history() {
        let mut graph = Graph::new();
        let node = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        graph
            .set_node_attr(node, "score".into(), AttrValue::Int(1))
            .unwrap();
        let _commit2 = graph.commit("c2".into(), "tester".into()).unwrap();
        graph
            .set_node_attr(node, "score".into(), AttrValue::Int(3))
            .unwrap();
        let commit3 = graph.commit("c3".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes: HashSet<NodeId> = [node].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let index = graph_rc.borrow().build_temporal_index().unwrap();
        let history = index.node_attr_history(node, &"score".to_string(), commit1, commit3);
        assert_eq!(history.len(), 2);

        let mut vars = StepVariables::default();
        vars.set_temporal_index("temporal_index", Arc::new(index));
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx =
            Context::with_temporal_scope(TemporalScope::with_window(commit3, commit1, commit3));

        let step =
            WindowAggregateStep::new("score", AggregateFunction::Sum, "agg", "temporal_index");
        step.apply(&mut ctx, &mut scope).unwrap();

        let map = scope.variables().node_map("agg").unwrap();
        assert_eq!(map.get(&node), Some(&AlgorithmParamValue::Float(4.0)));
    }

    #[test]
    fn temporal_filter_created_after_filters_nodes() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        let b = graph.add_node();
        let _commit2 = graph.commit("c2".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx = Context::new();

        let step = TemporalFilterStep::new(TemporalPredicate::CreatedAfter(commit1), "filtered");
        step.apply(&mut ctx, &mut scope).unwrap();

        let map = scope.variables().node_map("filtered").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(0)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(1)));
    }

    #[test]
    fn mark_changed_nodes_marks_created_nodes() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        let b = graph.add_node();
        let commit2 = graph.commit("c2".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx =
            Context::with_temporal_scope(TemporalScope::with_window(commit2, commit1, commit2));

        let step = MarkChangedNodesStep::new("changed", Some("created".into()));
        step.apply(&mut ctx, &mut scope).unwrap();

        let map = scope.variables().node_map("changed").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(1)));
    }

    #[test]
    fn snapshot_at_step_stores_snapshot() {
        let mut graph = Graph::new();
        let node = graph.add_node();
        let commit = graph.commit("c1".into(), "tester".into()).unwrap();
        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes: HashSet<NodeId> = [node].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx = Context::new();

        let step = SnapshotAtStep::new_at_commit(commit, "snapshot");
        step.apply(&mut ctx, &mut scope).unwrap();

        let snapshot = scope.variables().snapshot("snapshot").unwrap();
        assert_eq!(snapshot.lineage().commit_id, commit);
    }

    #[test]
    fn temporal_window_step_marks_nodes_in_range() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let commit1 = graph.commit("c1".into(), "tester".into()).unwrap();
        let b = graph.add_node();
        let _commit2 = graph.commit("c2".into(), "tester".into()).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(graph_rc.clone(), nodes, "test".into()).unwrap();

        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let mut ctx = Context::new();

        let step = TemporalWindowStep::new(commit1, commit1, "windowed");
        step.apply(&mut ctx, &mut scope).unwrap();

        let map = scope.variables().node_map("windowed").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(0)));
    }

    #[test]
    fn filter_nodes_by_attr_gt() {
        use crate::algorithms::steps::filtering::{FilterNodesByAttrStep, Predicate};

        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        graph
            .set_node_attr(a, AttrName::from("score"), AttrValue::Int(10))
            .unwrap();
        graph
            .set_node_attr(b, AttrName::from("score"), AttrValue::Int(20))
            .unwrap();

        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);

        let predicate = Predicate::Gt {
            value: AlgorithmParamValue::Int(15),
        };
        let step = FilterNodesByAttrStep::new("score", predicate, "filtered");
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("filtered").unwrap();
        assert_eq!(result.len(), 1); // Only node b should pass
        assert!(result.contains_key(&b));
        assert!(!result.contains_key(&a));
    }

    #[test]
    fn top_k_selects_highest() {
        use crate::algorithms::steps::filtering::{SortOrder, TopKStep};

        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Float(5.0));
        map.insert(nodes[1], AlgorithmParamValue::Float(10.0));
        vars.set_node_map("scores", map);

        let mut scope = StepScope::new(&sg, &mut vars);
        let step = TopKStep::new("scores", 1, "top", SortOrder::Descending);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("top").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains_key(&nodes[1])); // Highest score
    }

    #[test]
    fn sample_nodes_with_fraction() {
        use crate::algorithms::steps::sampling::{SampleNodesStep, SampleSpec};

        let (sg, _nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);

        // Sample 50% of nodes
        let spec = SampleSpec::Fraction { fraction: 0.5 };
        let step = SampleNodesStep::new(spec, Some(42), "sampled");
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("sampled").unwrap();
        // Should sample roughly half (with small graph, might not be exact)
        assert!(!result.is_empty());
        assert!(result.len() <= sg.node_count());
    }

    #[test]
    fn sample_nodes_with_count() {
        use crate::algorithms::steps::sampling::{SampleNodesStep, SampleSpec};

        let (sg, _nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);

        // Sample exactly 1 node
        let spec = SampleSpec::Count { count: 1 };
        let step = SampleNodesStep::new(spec, Some(42), "sampled");
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("sampled").unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn sample_nodes_reproducibility() {
        use crate::algorithms::steps::sampling::{SampleNodesStep, SampleSpec};

        let (sg, _nodes) = sample_subgraph();

        // First sample
        let mut vars1 = StepVariables::default();
        let mut scope1 = StepScope::new(&sg, &mut vars1);
        let spec1 = SampleSpec::Fraction { fraction: 0.5 };
        let step1 = SampleNodesStep::new(spec1, Some(123), "sampled");
        step1.apply(&mut Context::new(), &mut scope1).unwrap();
        let result1 = scope1.variables().node_map("sampled").unwrap();
        let keys1: Vec<NodeId> = result1.keys().copied().collect();

        // Second sample with same seed
        let mut vars2 = StepVariables::default();
        let mut scope2 = StepScope::new(&sg, &mut vars2);
        let spec2 = SampleSpec::Fraction { fraction: 0.5 };
        let step2 = SampleNodesStep::new(spec2, Some(123), "sampled");
        step2.apply(&mut Context::new(), &mut scope2).unwrap();
        let result2 = scope2.variables().node_map("sampled").unwrap();
        let mut keys2: Vec<NodeId> = result2.keys().copied().collect();

        // Should be identical
        keys2.sort();
        let mut sorted_keys1 = keys1.clone();
        sorted_keys1.sort();
        assert_eq!(sorted_keys1, keys2);
    }

    #[test]
    fn reservoir_sample_from_map() {
        use crate::algorithms::steps::sampling::{EntityType, ReservoirSampleStep};

        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();

        // Create a source map with 2 nodes
        let mut map = HashMap::new();
        map.insert(nodes[0], AlgorithmParamValue::Int(1));
        map.insert(nodes[1], AlgorithmParamValue::Int(2));
        vars.set_node_map("source", map);

        let mut scope = StepScope::new(&sg, &mut vars);
        let step = ReservoirSampleStep::new("source", 1, Some(42), "sampled", EntityType::Nodes);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let result = scope.variables().node_map("sampled").unwrap();
        assert_eq!(result.len(), 1);
        // Should be one of the source nodes
        assert!(result.contains_key(&nodes[0]) || result.contains_key(&nodes[1]));
    }
}
