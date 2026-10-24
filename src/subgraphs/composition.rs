//! Evaluate a whole composition before allocating entities or retained group contents.
use crate::api::graph::{composition::MetaComposition, Graph};
use crate::subgraphs::hierarchical::AggregationFunction;
use crate::traits::subgraph_operations::{
    AggregationSpec, EdgeAggregationConfig, ExternalEdgeStrategy, MetaEdgeStrategy, NodeStrategy,
};
use crate::traits::SubgraphOperations;
use crate::{AttrName, AttrValue, BatchNodeRef, GraphError, GraphResult, NodeId};
use std::collections::{BTreeMap, HashMap, HashSet};

type EdgeRow = (NodeId, NodeId, HashMap<AttrName, AttrValue>);

pub(crate) fn compose<T: SubgraphOperations + ?Sized>(
    selection: &T,
    specs: &[AggregationSpec],
    config: &EdgeAggregationConfig,
    lenient: bool,
) -> GraphResult<NodeId> {
    let binding = selection.graph_ref();
    let mut graph = binding.borrow_mut();
    let mut nodes: Vec<_> = selection.node_set().iter().copied().collect();
    nodes.sort_unstable();
    for &node in &nodes {
        if !graph.contains_node(node) {
            return Err(GraphError::InvalidInput(format!(
                "Composition node {node} is inactive"
            )));
        }
    }
    for &edge in selection.edge_set() {
        let (source, target) = graph.edge_endpoints(edge)?;
        if !graph.contains_edge(edge)
            || !selection.node_set().contains(&source)
            || !selection.node_set().contains(&target)
        {
            return Err(GraphError::InvalidInput(
                "Composition edges must be active and internal to the selection".into(),
            ));
        }
    }
    let mut attrs = HashMap::new();
    for spec in specs {
        if matches!(
            spec.target_attr.as_str(),
            "entity_type" | "external_id" | "contains_subgraph"
        ) {
            return Err(GraphError::InvalidInput(format!(
                "{} is managed by graph identity",
                spec.target_attr
            )));
        }
        let function = AggregationFunction::from_string(&spec.function)?;
        let value = if matches!(function, AggregationFunction::Count) {
            AttrValue::Int(
                i64::try_from(nodes.len())
                    .map_err(|_| GraphError::InvalidInput("Node count exceeds i64".into()))?,
            )
        } else {
            let mut values = Vec::new();
            if let Some(source) = &spec.source_attr {
                for &node in &nodes {
                    if let Some(value) = graph.get_node_attr(node, source)? {
                        values.push(value);
                    }
                }
            }
            if !values.is_empty() {
                function.aggregate(&values)?
            } else if let Some(default) = &spec.default_value {
                default.clone()
            } else if lenient {
                match function {
                    AggregationFunction::Sum => AttrValue::Int(0),
                    AggregationFunction::Mean
                    | AggregationFunction::Min
                    | AggregationFunction::Max => AttrValue::Float64(0.0),
                    _ => AttrValue::Text(String::new()),
                }
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Attribute {:?} has no observations and no default",
                    spec.source_attr
                )));
            }
        };
        if attrs.insert(spec.target_attr.clone(), value).is_some() {
            return Err(GraphError::InvalidInput(format!(
                "Duplicate aggregation target {}",
                spec.target_attr
            )));
        }
    }
    let meta = graph.pool().statistics().node_count;
    let boundary = boundary_edges(&graph, selection.node_set(), meta, config)?
        .into_iter()
        .map(|(source, target, attrs)| {
            let resolve = |id| {
                if id == meta {
                    BatchNodeRef::Added(0)
                } else {
                    BatchNodeRef::Existing(id)
                }
            };
            (resolve(source), resolve(target), attrs)
        })
        .collect();
    graph.apply_meta_composition(MetaComposition {
        nodes: selection.node_set().clone(),
        edges: selection.edge_set().clone(),
        kind: selection.entity_type().into(),
        attrs,
        boundary,
        remove_originals: matches!(config.node_strategy, NodeStrategy::Collapse),
    })
}

pub(crate) fn boundary_edges(
    graph: &Graph,
    nodes: &HashSet<NodeId>,
    meta: NodeId,
    config: &EdgeAggregationConfig,
) -> GraphResult<Vec<EdgeRow>> {
    if config.edge_to_external == ExternalEdgeStrategy::None {
        return Ok(Vec::new());
    }
    if config.edge_to_meta == MetaEdgeStrategy::Explicit {
        return Err(GraphError::InvalidInput(
            "Explicit meta-edge selection requires a selector; use Auto or None".into(),
        ));
    }
    let mut groups: BTreeMap<(NodeId, NodeId), Vec<HashMap<AttrName, AttrValue>>> = BTreeMap::new();
    let mut edges = graph.edge_ids();
    edges.sort_unstable();
    for edge in edges {
        let (source, target) = graph.edge_endpoints(edge)?;
        let (inside_source, inside_target) = (nodes.contains(&source), nodes.contains(&target));
        if inside_source == inside_target {
            continue;
        }
        let external = if inside_source { target } else { source };
        if config.edge_to_meta == MetaEdgeStrategy::None && graph.is_meta_node(external) {
            continue;
        }
        let endpoints = if graph.is_directed() && !inside_source {
            (external, meta)
        } else {
            (meta, external)
        };
        groups
            .entry(endpoints)
            .or_default()
            .push(graph.get_edge_attrs(edge)?);
    }
    let mut rows = Vec::new();
    for ((source, target), originals) in groups {
        if originals.len() < config.min_edge_count as usize {
            continue;
        }
        let count = i64::try_from(originals.len())
            .map_err(|_| GraphError::InvalidInput("Edge count exceeds i64".into()))?;
        let mut outputs = match config.edge_to_external {
            ExternalEdgeStrategy::Copy => originals,
            ExternalEdgeStrategy::Count => vec![HashMap::new()],
            ExternalEdgeStrategy::Aggregate => {
                let mut columns: BTreeMap<AttrName, Vec<AttrValue>> = BTreeMap::new();
                for original in originals {
                    for (name, value) in original {
                        columns.entry(name).or_default().push(value);
                    }
                }
                let mut attrs = HashMap::new();
                for (name, values) in columns {
                    let function = config
                        .edge_aggregation
                        .get(&name)
                        .unwrap_or(&config.default_aggregation);
                    attrs.insert(name, function.aggregate(&values)?);
                }
                vec![attrs]
            }
            ExternalEdgeStrategy::None => unreachable!(),
        };
        for attrs in &mut outputs {
            decorate_edge(
                attrs,
                source,
                target,
                if config.edge_to_external == ExternalEdgeStrategy::Copy {
                    1
                } else {
                    count
                },
                config,
            )?;
        }
        rows.extend(outputs.into_iter().map(|attrs| (source, target, attrs)));
    }
    Ok(rows)
}

/// Attach generated metadata only after validating the staged source attributes.
pub(crate) fn decorate_edge(
    attrs: &mut HashMap<AttrName, AttrValue>,
    source: NodeId,
    target: NodeId,
    count: i64,
    config: &EdgeAggregationConfig,
) -> GraphResult<()> {
    if attrs.contains_key("external_id") {
        return Err(GraphError::InvalidInput(
            "Meta-edges cannot duplicate managed external_id; use explicit upserts".into(),
        ));
    }
    if config.include_edge_count || config.edge_to_external == ExternalEdgeStrategy::Count {
        attrs.insert("edge_count".into(), AttrValue::Int(count));
    }
    if config.mark_entity_type {
        attrs.insert("entity_type".into(), AttrValue::Text("meta".into()));
    }
    attrs.insert(
        "source".into(),
        AttrValue::Int(
            i64::try_from(source)
                .map_err(|_| GraphError::InvalidInput("Source ID exceeds i64".into()))?,
        ),
    );
    attrs.insert(
        "target".into(),
        AttrValue::Int(
            i64::try_from(target)
                .map_err(|_| GraphError::InvalidInput("Target ID exceeds i64".into()))?,
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subgraphs::{MetaNodePlan, Subgraph};
    use crate::traits::NodeOperations;
    use std::{cell::RefCell, rc::Rc};

    fn fixture(publishing: bool) -> (Rc<RefCell<Graph>>, Subgraph) {
        let mut graph = if publishing {
            Graph::new_publishing()
        } else {
            Graph::new()
        };
        let nodes = graph.insert_nodes(3).expect("nodes");
        graph
            .set_node_attr(nodes[0], "x".into(), AttrValue::Int(2))
            .expect("x");
        graph
            .set_node_attr(nodes[1], "x".into(), AttrValue::Int(4))
            .expect("x");
        graph.add_edge(nodes[0], nodes[2]).expect("boundary");
        let shared = Rc::new(RefCell::new(graph));
        let selection = Subgraph::new(
            shared.clone(),
            [0, 1].into_iter().collect(),
            HashSet::new(),
            "test".into(),
        );
        (shared, selection)
    }

    #[test]
    fn rejected_compositions_do_not_allocate_entities_or_store_groups() {
        let (shared, selection) = fixture(true);
        let state = shared.borrow().current_state_id();
        let groups = shared.borrow().pool().subgraph_count();
        for target in ["entity_type", "external_id", "contains_subgraph", "missing"] {
            let plan = MetaNodePlan::new(selection.node_set().clone(), HashSet::new())
                .with_node_agg(target.into(), "sum".into(), Some("missing".into()));
            assert!(plan.add_to_graph(&selection).is_err());
            let graph = shared.borrow();
            assert_eq!(graph.current_state_id(), state);
            assert_eq!(graph.pool().subgraph_count(), groups);
            assert_eq!(graph.pool().statistics().node_count, 3);
            assert_eq!(graph.pool().statistics().edge_count, 1);
        }
        let invalid = Subgraph::new(
            shared.clone(),
            [0].into_iter().collect(),
            [0].into_iter().collect(),
            "bad".into(),
        );
        assert!(compose(&invalid, &[], &EdgeAggregationConfig::default(), false).is_err());
        let mut plan = MetaNodePlan::new(HashSet::new(), HashSet::new());
        assert!(plan.add_to_graph(&selection).is_err());
        plan.source_nodes = selection.node_set().clone();
        let meta = plan.add_to_graph(&selection).expect("valid composition");
        assert_eq!(meta.node_id(), 3);
        assert_eq!(shared.borrow().current_state_id(), state + 1);
    }

    #[test]
    fn late_boundary_failure_and_stale_selection_leave_no_group_or_ids() {
        let (shared, selection) = fixture(true);
        shared
            .borrow_mut()
            .set_edge_attr(0, "w".into(), AttrValue::Text("bad".into()))
            .expect("w");
        let state = shared.borrow().current_state_id();
        let groups = shared.borrow().pool().subgraph_count();
        let config = EdgeAggregationConfig {
            edge_aggregation: [(
                "w".into(),
                crate::traits::subgraph_operations::EdgeAggregationFunction::Sum,
            )]
            .into_iter()
            .collect(),
            node_strategy: NodeStrategy::Collapse,
            ..Default::default()
        };
        assert!(compose(&selection, &[], &config, false).is_err());
        assert_eq!(shared.borrow().current_state_id(), state);
        assert_eq!(shared.borrow().pool().subgraph_count(), groups);
        assert_eq!(shared.borrow().pool().statistics().node_count, 3);
        assert_eq!(shared.borrow().pool().statistics().edge_count, 1);
        shared.borrow_mut().remove_node(0).expect("remove");
        let state = shared.borrow().current_state_id();
        assert!(compose(&selection, &[], &EdgeAggregationConfig::default(), true).is_err());
        assert_eq!(shared.borrow().current_state_id(), state);
        assert_eq!(shared.borrow().pool().subgraph_count(), groups);
    }

    #[test]
    fn meta_boundary_policy_and_threshold_are_explicit() {
        let (shared, selection) = fixture(true);
        let group = shared
            .borrow_mut()
            .pool_mut()
            .store_subgraph([2].into_iter().collect(), HashSet::new(), "outer".into())
            .expect("group");
        let outside = shared
            .borrow_mut()
            .create_meta_node(group)
            .expect("outside group");
        shared.borrow_mut().add_edge(0, outside).expect("to meta");
        let mut config = EdgeAggregationConfig {
            edge_to_meta: MetaEdgeStrategy::None,
            ..Default::default()
        };
        let graph = shared.borrow();
        let rows =
            boundary_edges(&graph, selection.node_set(), 10, &config).expect("no meta boundary");
        assert_eq!(rows.len(), 1);
        config.min_edge_count = 2;
        assert!(boundary_edges(&graph, selection.node_set(), 10, &config)
            .expect("threshold")
            .is_empty());
        config.edge_to_meta = MetaEdgeStrategy::Explicit;
        assert!(boundary_edges(&graph, selection.node_set(), 10, &config).is_err());
    }

    #[test]
    fn explicit_defaults_only_apply_when_observations_are_absent() {
        let (shared, selection) = fixture(false);
        let meta = selection
            .collapse_to_node_with_defaults(
                [("x".into(), "sum".into()), ("absent".into(), "sum".into())]
                    .into_iter()
                    .collect(),
                [
                    ("x".into(), AttrValue::Int(99)),
                    ("absent".into(), AttrValue::Int(7)),
                ]
                .into_iter()
                .collect(),
            )
            .expect("legacy defaults");
        let graph = shared.borrow();
        assert_eq!(
            graph.get_node_attr(meta, &"x".into()).expect("x"),
            Some(AttrValue::Int(6))
        );
        assert_eq!(
            graph.get_node_attr(meta, &"absent".into()).expect("absent"),
            Some(AttrValue::Int(7))
        );
        assert_eq!(graph.current_state_id(), 0);
        assert!(graph.has_uncommitted_changes());
    }
}
