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
    compose_with_label(selection, specs, config, lenient, None)
}

pub(crate) fn compose_with_label<T: SubgraphOperations + ?Sized>(
    selection: &T,
    specs: &[AggregationSpec],
    config: &EdgeAggregationConfig,
    lenient: bool,
    semantic_kind: Option<&str>,
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
    if let Some(kind) = semantic_kind {
        if kind.trim().is_empty() {
            return Err(GraphError::InvalidInput(
                "Composition kind must be nonempty".into(),
            ));
        }
        if kind != "meta" {
            let labels = attrs
                .entry(crate::LABELS.into())
                .or_insert_with(|| AttrValue::TextVec(Vec::new()));
            let AttrValue::TextVec(labels) = labels else {
                return Err(GraphError::InvalidInput(
                    "Composition labels must be a text vector".into(),
                ));
            };
            if !labels.iter().any(|label| label == kind) {
                labels.push(kind.to_string());
            }
        }
    }
    let meta = graph.pool().statistics().node_count;
    let staged = if config.edge_to_external == ExternalEdgeStrategy::Contract {
        contraction_edges(&graph, selection.node_set(), selection.edge_set(), config)?
    } else {
        boundary_edges(&graph, selection.node_set(), meta, config)?
    };
    let boundary = staged
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
    validate_meta_targets(graph, nodes, config)?;
    if config.edge_to_external == ExternalEdgeStrategy::None {
        return Ok(Vec::new());
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
        if graph.is_meta_node(external) {
            match &config.edge_to_meta {
                MetaEdgeStrategy::None => continue,
                MetaEdgeStrategy::Selected(targets) if !targets.contains(&external) => continue,
                _ => {}
            }
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
            ExternalEdgeStrategy::Contract => {
                return Err(GraphError::InvalidInput(
                    "Contraction requires selected internal edges".into(),
                ))
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

fn validate_meta_targets(
    graph: &Graph,
    nodes: &HashSet<NodeId>,
    config: &EdgeAggregationConfig,
) -> GraphResult<()> {
    match &config.edge_to_meta {
        MetaEdgeStrategy::Explicit => {
            return Err(GraphError::InvalidInput(
                "Explicit meta-edge selection requires targets; use with_meta_targets".into(),
            ))
        }
        MetaEdgeStrategy::Selected(targets) => {
            let mut ids: Vec<_> = targets.iter().copied().collect();
            ids.sort_unstable();
            for id in ids {
                if nodes.contains(&id) || !graph.contains_node(id) || !graph.is_meta_node(id) {
                    return Err(GraphError::InvalidInput(format!(
                        "Explicit meta target {id} must be an active external meta-node"
                    )));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn meta_target_allowed(graph: &Graph, node: NodeId, config: &EdgeAggregationConfig) -> bool {
    if !graph.is_meta_node(node) {
        return true;
    }
    match &config.edge_to_meta {
        MetaEdgeStrategy::None => false,
        MetaEdgeStrategy::Selected(targets) => targets.contains(&node),
        _ => true,
    }
}

fn reachable(
    seeds: impl Iterator<Item = NodeId>,
    adjacency: &HashMap<NodeId, Vec<NodeId>>,
) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut stack = Vec::new();
    for node in seeds {
        if seen.insert(node) {
            stack.push(node);
        }
    }
    while let Some(node) = stack.pop() {
        if let Some(row) = adjacency.get(&node) {
            for &target in row {
                if seen.insert(target) {
                    stack.push(target);
                }
            }
        }
    }
    seen
}

/// Reachability contraction with distinct original-edge corridor aggregation.
/// Directed outputs preserve orientation; undirected outputs use increasing IDs.
/// Internal traversal is bounded by the exact selected edge set, never the owner CSR.
pub(crate) fn contraction_edges(
    graph: &Graph,
    nodes: &HashSet<NodeId>,
    internal: &HashSet<crate::EdgeId>,
    config: &EdgeAggregationConfig,
) -> GraphResult<Vec<EdgeRow>> {
    validate_meta_targets(graph, nodes, config)?;
    if nodes.iter().any(|&node| !graph.has_node(node)) {
        return Err(GraphError::InvalidInput(
            "Contraction nodes must be active".into(),
        ));
    }
    let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut backward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut inside = Vec::new();
    let mut ids: Vec<_> = internal.iter().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let (source, target) = graph.edge_endpoints(id)?;
        if !graph.has_edge(id) || !nodes.contains(&source) || !nodes.contains(&target) {
            return Err(GraphError::InvalidInput(
                "Contraction internal edges must be active and selected".into(),
            ));
        }
        forward.entry(source).or_default().push(target);
        backward.entry(target).or_default().push(source);
        if graph.is_undirected() {
            forward.entry(target).or_default().push(source);
            backward.entry(source).or_default().push(target);
        }
        inside.push((id, source, target));
    }
    let mut incoming: BTreeMap<NodeId, Vec<(NodeId, crate::EdgeId)>> = BTreeMap::new();
    let mut outgoing: BTreeMap<NodeId, Vec<(NodeId, crate::EdgeId)>> = BTreeMap::new();
    let mut all = graph.edge_ids();
    all.sort_unstable();
    for edge in all {
        let (source, target) = graph.edge_endpoints(edge)?;
        match (nodes.contains(&source), nodes.contains(&target)) {
            (false, true) if meta_target_allowed(graph, source, config) => {
                incoming.entry(source).or_default().push((target, edge));
                if graph.is_undirected() {
                    outgoing.entry(source).or_default().push((target, edge));
                }
            }
            (true, false) if meta_target_allowed(graph, target, config) => {
                outgoing.entry(target).or_default().push((source, edge));
                if graph.is_undirected() {
                    incoming.entry(target).or_default().push((source, edge));
                }
            }
            _ => {}
        }
    }
    let mut rows = Vec::new();
    for (&source, entries) in &incoming {
        let from_source = reachable(entries.iter().map(|&(node, _)| node), &forward);
        for (&target, exits) in &outgoing {
            if source == target || (graph.is_undirected() && source > target) {
                continue;
            }
            if !exits.iter().any(|(node, _)| from_source.contains(node)) {
                continue;
            }
            let to_target = reachable(exits.iter().map(|&(node, _)| node), &backward);
            let mut contributing = std::collections::BTreeSet::new();
            for &(node, edge) in entries {
                if to_target.contains(&node) {
                    contributing.insert(edge);
                }
            }
            for &(node, edge) in exits {
                if from_source.contains(&node) {
                    contributing.insert(edge);
                }
            }
            for &(edge, left, right) in &inside {
                if (from_source.contains(&left) && to_target.contains(&right))
                    || (graph.is_undirected()
                        && from_source.contains(&right)
                        && to_target.contains(&left))
                {
                    contributing.insert(edge);
                }
            }
            if contributing.len() < config.min_edge_count as usize {
                continue;
            }
            let count = i64::try_from(contributing.len()).map_err(|_| {
                GraphError::InvalidInput("Contraction edge count exceeds i64".into())
            })?;
            let mut columns: BTreeMap<AttrName, Vec<AttrValue>> = BTreeMap::new();
            for edge in contributing {
                for (name, value) in graph.get_edge_attrs(edge)? {
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
            decorate_edge(&mut attrs, source, target, count, config)?;
            rows.push((source, target, attrs));
        }
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
            Graph::new_manual()
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
    fn contraction_matches_independent_small_graph_reachability_and_corridor_counts() {
        use crate::storage::numeric::scalar::attribute_number;
        for directed in [false, true] {
            let pairs = if directed {
                vec![
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (2, 0),
                    (1, 3),
                    (2, 3),
                    (3, 1),
                    (3, 2),
                    (1, 2),
                    (2, 1),
                    (1, 1),
                ]
            } else {
                vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (1, 1)]
            };
            for bits in 0..(1usize << pairs.len()) {
                let mut graph = if directed {
                    Graph::new_directed_manual()
                } else {
                    Graph::new_manual()
                };
                graph.add_nodes(4).unwrap();
                let mut internal = HashSet::new();
                let mut originals = Vec::new();
                let mut closure = [[false; 4]; 4];
                closure[1][1] = true;
                closure[2][2] = true;
                for (bit, &(left, right)) in pairs.iter().enumerate() {
                    if bits & (1 << bit) == 0 {
                        continue;
                    }
                    let id = graph.add_edge(left, right).unwrap();
                    graph
                        .set_edge_attr(id, "w".into(), AttrValue::Int(1))
                        .unwrap();
                    if [1, 2].contains(&left) && [1, 2].contains(&right) {
                        internal.insert(id);
                    }
                    originals.push((id, left, right));
                    closure[left][right] = true;
                    if !directed {
                        closure[right][left] = true;
                    }
                }
                // Independent Boolean transitive closure permits only selected intermediates.
                for mid in [1, 2] {
                    for left in 0..4 {
                        for right in 0..4 {
                            closure[left][right] |= closure[left][mid] && closure[mid][right];
                        }
                    }
                }
                let config = EdgeAggregationConfig {
                    edge_to_external: ExternalEdgeStrategy::Contract,
                    edge_aggregation: [(
                        "w".into(),
                        crate::traits::subgraph_operations::EdgeAggregationFunction::Sum,
                    )]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                };
                let rows =
                    contraction_edges(&graph, &[1, 2].into_iter().collect(), &internal, &config)
                        .unwrap();
                let expected: Vec<_> = [(0, 3), (3, 0)]
                    .into_iter()
                    .filter(|&(left, right)| (directed || left < right) && closure[left][right])
                    .collect();
                assert_eq!(
                    rows.iter()
                        .map(|(left, right, _)| (*left, *right))
                        .collect::<Vec<_>>(),
                    expected,
                    "directed={directed} mask={bits}"
                );
                for (source, target, attrs) in rows {
                    let participates = |left: usize, right: usize| {
                        if [1, 2].contains(&left) && [1, 2].contains(&right) {
                            closure[source][left] && closure[right][target]
                        } else if left == source && [1, 2].contains(&right) {
                            closure[right][target]
                        } else {
                            right == target && [1, 2].contains(&left) && closure[source][left]
                        }
                    };
                    let count = originals
                        .iter()
                        .filter(|&&(_, left, right)| {
                            participates(left, right) || (!directed && participates(right, left))
                        })
                        .count();
                    assert_eq!(attrs["edge_count"], AttrValue::Int(count as i64));
                    assert_eq!(
                        attribute_number(&attrs["w"]).unwrap().as_f64(),
                        count as f64
                    );
                }
            }
        }
    }

    #[test]
    fn contraction_does_not_use_owner_internal_edges_excluded_from_selection() {
        let mut graph = Graph::new_directed_manual();
        graph.add_nodes(4).unwrap();
        graph.add_edge(0, 1).unwrap();
        let internal = graph.add_edge(1, 2).unwrap();
        graph.add_edge(2, 3).unwrap();
        let config = EdgeAggregationConfig {
            edge_to_external: ExternalEdgeStrategy::Contract,
            ..Default::default()
        };
        let nodes = [1, 2].into_iter().collect();
        assert!(contraction_edges(&graph, &nodes, &HashSet::new(), &config)
            .unwrap()
            .is_empty());
        assert_eq!(
            contraction_edges(&graph, &nodes, &[internal].into_iter().collect(), &config)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn preset_and_explicit_semantic_kinds_preserve_meta_identity_and_one_acceptance() {
        use crate::subgraphs::composer::ComposerOptions;
        for (preset, explicit, expected) in [
            (Some("social_network"), None, "community"),
            (Some("org_hierarchy"), None, "department"),
            (Some("social_network"), Some("custom"), "custom"),
        ] {
            let (shared, selection) = fixture(true);
            let before = shared.borrow().current_state_id();
            let plan = selection
                .collapse_with_options(ComposerOptions {
                    preset: preset.map(str::to_string),
                    entity_type: explicit.map(str::to_string),
                    ..Default::default()
                })
                .unwrap();
            let meta = plan.add_to_graph(&selection).unwrap();
            let graph = shared.borrow();
            assert_eq!(graph.current_state_id(), before + 1);
            assert!(graph.is_meta_node(meta.id()));
            assert_eq!(
                graph.node_labels(meta.id()).unwrap(),
                vec![expected.to_string()]
            );
            assert_eq!(
                graph
                    .snapshot_at_commit(before + 1)
                    .unwrap()
                    .node_attr(meta.id(), &crate::LABELS.into()),
                Some(AttrValue::TextVec(vec![expected.to_string()]))
            );
        }
    }

    #[test]
    fn explicit_options_override_presets_without_losing_omitted_defaults() {
        use crate::subgraphs::composer::{ComposerOptions, EdgeStrategy};
        let (_, selection) = fixture(true);
        let plan = selection
            .collapse_with_options(ComposerOptions {
                preset: Some("org_hierarchy".into()),
                ..Default::default()
            })
            .unwrap();
        assert!(!plan.include_edge_count);
        assert_eq!(plan.entity_type, "department");
        let plan = selection
            .collapse_with_options(ComposerOptions {
                preset: Some("org_hierarchy".into()),
                include_edge_count: Some(true),
                edge_strategy: Some(EdgeStrategy::KeepExternal),
                entity_type: Some("meta".into()),
                ..Default::default()
            })
            .unwrap();
        assert!(plan.include_edge_count);
        assert_eq!(plan.entity_type, "meta");
        assert_eq!(plan.edge_strategy, EdgeStrategy::KeepExternal);
    }

    #[test]
    fn explicit_meta_targets_validate_and_filter_without_creating_edges_to_unselected_metas() {
        let (shared, selection) = fixture(true);
        let stored = shared
            .borrow_mut()
            .pool_mut()
            .store_subgraph([2].into_iter().collect(), HashSet::new(), "outside".into())
            .unwrap();
        let outside = shared.borrow_mut().create_meta_node(stored).unwrap();
        shared.borrow_mut().add_edge(0, outside).unwrap();
        let graph = shared.borrow();
        let selected =
            EdgeAggregationConfig::default().with_meta_targets([outside].into_iter().collect());
        assert_eq!(
            boundary_edges(&graph, selection.node_set(), 99, &selected)
                .unwrap()
                .len(),
            2
        );
        let excluded = EdgeAggregationConfig::default().with_meta_targets(HashSet::new());
        let rows = boundary_edges(&graph, selection.node_set(), 99, &excluded).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].0, rows[0].1), (99, 2));
        for invalid in [0, 2, 999] {
            let config =
                EdgeAggregationConfig::default().with_meta_targets([invalid].into_iter().collect());
            assert!(boundary_edges(&graph, selection.node_set(), 99, &config).is_err());
        }
        assert!(MetaEdgeStrategy::selected([outside, outside]).is_err());
    }

    #[test]
    fn invalid_kinds_and_selector_inputs_do_not_allocate_or_publish() {
        let (shared, selection) = fixture(true);
        let state = shared.borrow().current_state_id();
        let groups = shared.borrow().pool().subgraph_count();
        let plan = MetaNodePlan::new(selection.node_set().clone(), selection.edge_set().clone());
        assert!(plan
            .clone()
            .with_entity_type(" ".into())
            .add_to_graph(&selection)
            .is_err());
        for targets in [vec![2], vec![0, 0]] {
            assert!(plan
                .execute_configured(&selection, false, Some(&targets))
                .is_err());
        }
        let graph = shared.borrow();
        assert_eq!(graph.current_state_id(), state);
        assert_eq!(graph.pool().subgraph_count(), groups);
        assert_eq!(graph.pool().statistics().node_count, 3);
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
