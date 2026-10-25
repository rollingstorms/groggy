//! Checked domain-local identity planning and lossless graph-table merges.
use super::{
    BaseArray, BaseTable, ConflictResolution as Strategy, EdgesTable, GraphError, GraphResult,
    GraphTable, HashMap, HashSet, MergeOptions, NodesTable, Path, Table, ValidationPolicy,
    ValidationStrictness,
};
use crate::{storage::table::structural, AttrValue as V};
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn id_value(id: usize) -> GraphResult<V> {
    i64::try_from(id)
        .map(V::Int)
        .map_err(|_| invalid("Merged ID exceeds table integer range"))
}
struct Domain {
    nodes: Vec<usize>,
    edges: Vec<usize>,
    sources: Vec<usize>,
    targets: Vec<usize>,
}
fn ids(table: &BaseTable, name: &str) -> GraphResult<Vec<usize>> {
    let ids = structural::ids(table, name)?;
    let mut seen = HashSet::new();
    for &id in &ids {
        if !seen.insert(id) {
            return Err(invalid(format!("Duplicate {name} {id} within input table")));
        }
    }
    Ok(ids)
}
fn inspect(table: &GraphTable) -> GraphResult<Domain> {
    let nodes = ids(table.nodes.base(), "node_id")?;
    let edges = ids(table.edges.base(), "edge_id")?;
    let sources = structural::ids(table.edges.base(), "source")?;
    let targets = structural::ids(table.edges.base(), "target")?;
    let node_set: HashSet<_> = nodes.iter().copied().collect();
    for (&source, &target) in sources.iter().zip(&targets) {
        if !node_set.contains(&source) || !node_set.contains(&target) {
            return Err(invalid("Edge endpoint absent from its input domain"));
        }
    }
    for base in [table.nodes.base(), table.edges.base()] {
        let mut seen = HashSet::new();
        for name in base.column_names() {
            if !seen.insert(name) {
                return Err(invalid("Duplicate input column name"));
            }
            structural::column(base, name)?;
        }
    }
    Ok(Domain {
        nodes,
        edges,
        sources,
        targets,
    })
}
fn mappings(
    domains: &[Domain],
    nodes: bool,
    strategy: &Strategy,
) -> GraphResult<Vec<HashMap<usize, usize>>> {
    let reserved: HashSet<_> = domains
        .iter()
        .flat_map(|d| if nodes { &d.nodes } else { &d.edges })
        .copied()
        .collect();
    let mut used = HashSet::new();
    let mut next = 0usize;
    let mut result = Vec::new();
    for domain in domains {
        let mut map = HashMap::new();
        for &id in if nodes { &domain.nodes } else { &domain.edges } {
            let mapped = if *strategy == Strategy::DomainPrefix {
                let id = used.len();
                id_value(id)?;
                used.insert(id);
                id
            } else if used.insert(id) {
                id
            } else {
                match strategy {
                    Strategy::Fail => {
                        return Err(invalid(format!(
                            "{} ID collision detected: {id}",
                            if nodes { "Node" } else { "Edge" }
                        )))
                    }
                    Strategy::AutoRemap => {
                        while reserved.contains(&next) || used.contains(&next) {
                            next = next
                                .checked_add(1)
                                .ok_or_else(|| invalid("Automatic ID allocation exhausted"))?;
                        }
                        id_value(next)?;
                        used.insert(next);
                        next
                    }
                    _ => id,
                }
            };
            map.insert(id, mapped);
        }
        result.push(map);
    }
    Ok(result)
}
fn merged_policy(tables: &[GraphTable]) -> GraphResult<ValidationPolicy> {
    let mut p = tables
        .first()
        .ok_or_else(|| invalid("Cannot merge empty list of tables"))?
        .policy
        .clone();
    for table in tables {
        let q = &table.policy;
        if q.strictness == ValidationStrictness::Strict
            || (q.strictness == ValidationStrictness::Standard
                && p.strictness == ValidationStrictness::Minimal)
        {
            p.strictness = q.strictness.clone();
        }
        p.edge_config.allow_self_loops &= q.edge_config.allow_self_loops;
        p.edge_config.allow_multi_edges &= q.edge_config.allow_multi_edges;
        p.validate_node_references |= q.validate_node_references;
        p.edge_config.validate_node_references |= q.edge_config.validate_node_references;
        p.auto_repair &= q.auto_repair;
    }
    Ok(p)
}
fn merged_component(
    tables: &[GraphTable],
    domains: &[Domain],
    node_maps: &[HashMap<usize, usize>],
    edge_maps: &[HashMap<usize, usize>],
    nodes: bool,
    strategy: &Strategy,
    names: Option<&[String]>,
) -> GraphResult<BaseTable> {
    let structural_names = if nodes {
        vec!["node_id".to_string()]
    } else {
        vec![
            "edge_id".to_string(),
            "source".to_string(),
            "target".to_string(),
        ]
    };
    let remapping = matches!(strategy, Strategy::DomainPrefix | Strategy::AutoRemap);
    let mut order = structural_names.clone();
    let mut seen: HashSet<_> = order.iter().cloned().collect();
    let mut origins = Vec::new();
    for table in tables {
        let base = if nodes {
            table.nodes.base()
        } else {
            table.edges.base()
        };
        origins.push(base);
        for name in base.column_names() {
            if remapping && matches!(name.as_str(), "_domain" | "_original_id") {
                return Err(invalid(format!(
                    "Reserved merge metadata column '{name}' already exists"
                )));
            }
            if seen.insert(name.clone()) {
                order.push(name.clone());
            }
        }
    }
    if remapping {
        order.extend(["_domain".into(), "_original_id".into()]);
    }
    // Columnar output buffers, indexed once per distinct entity ID.
    let mut columns: HashMap<String, Vec<V>> =
        order.iter().map(|n| (n.clone(), Vec::new())).collect();
    let mut positions = HashMap::new();
    let mut rows = 0;
    for (domain_index, table) in tables.iter().enumerate() {
        let base = if nodes {
            table.nodes.base()
        } else {
            table.edges.base()
        };
        let domain = &domains[domain_index];
        let maps = if nodes {
            &node_maps[domain_index]
        } else {
            &edge_maps[domain_index]
        };
        for (row, &old_id) in if nodes { &domain.nodes } else { &domain.edges }
            .iter()
            .enumerate()
        {
            let id = *maps
                .get(&old_id)
                .ok_or_else(|| invalid("Missing domain ID mapping"))?;
            let previous = positions.get(&id).copied();
            let position = if let Some(p) = previous {
                p
            } else {
                let p = rows;
                rows += 1;
                positions.insert(id, p);
                for values in columns.values_mut() {
                    values.push(V::Null);
                }
                p
            };
            let endpoints = if nodes {
                None
            } else {
                Some((
                    *node_maps[domain_index]
                        .get(&domain.sources[row])
                        .ok_or_else(|| invalid("Missing source mapping"))?,
                    *node_maps[domain_index]
                        .get(&domain.targets[row])
                        .ok_or_else(|| invalid("Missing target mapping"))?,
                ))
            };
            if previous.is_some() && *strategy == Strategy::MergeAttributes {
                if let Some((source, target)) = endpoints {
                    if columns.get("source").and_then(|c| c.get(position))
                        != Some(&id_value(source)?)
                        || columns.get("target").and_then(|c| c.get(position))
                            != Some(&id_value(target)?)
                    {
                        return Err(invalid(format!("Edge {id} endpoint conflict; choose keep_first, keep_second or remapping explicitly")));
                    }
                }
            }
            // Validate/remap every typed reference, including rows discarded by a keep policy.
            let mut prepared = Vec::new();
            for name in base.column_names() {
                if structural_names.contains(name) {
                    continue;
                }
                let value = structural::column(base, name)?
                    .get(row)
                    .cloned()
                    .ok_or_else(|| invalid("Missing merge property cell"))?;
                let value = crate::storage::value_references::remap(
                    value,
                    &node_maps[domain_index],
                    &edge_maps[domain_index],
                    "merge input domain",
                )
                .map_err(|e| invalid(format!("Domain {domain_index} {name} row {row}: {e}")))?;
                prepared.push((name, value));
            }
            if previous.is_some() && *strategy == Strategy::KeepFirst {
                continue;
            }
            if previous.is_some() && *strategy == Strategy::KeepSecond {
                for values in columns.values_mut() {
                    values[position] = V::Null;
                }
            }
            let id_name = if nodes { "node_id" } else { "edge_id" };
            set(&mut columns, id_name, position, id_value(id)?)?;
            if let Some((source, target)) = endpoints {
                set(&mut columns, "source", position, id_value(source)?)?;
                set(&mut columns, "target", position, id_value(target)?)?;
            }
            for (name, value) in prepared {
                set(&mut columns, name, position, value)?;
            }
            if remapping {
                set(
                    &mut columns,
                    "_domain",
                    position,
                    V::Text(
                        names
                            .map(|n| n[domain_index].clone())
                            .unwrap_or_else(|| format!("domain_{domain_index}")),
                    ),
                )?;
                set(&mut columns, "_original_id", position, id_value(old_id)?)?;
            }
        }
    }
    BaseTable::collection_result(
        columns
            .into_iter()
            .map(|(n, v)| (n, BaseArray::from_attr_values(v)))
            .collect(),
        order,
        rows,
        origins,
    )
}
fn set(columns: &mut HashMap<String, Vec<V>>, name: &str, row: usize, value: V) -> GraphResult<()> {
    let cell = columns
        .get_mut(name)
        .and_then(|c| c.get_mut(row))
        .ok_or_else(|| invalid("Invalid merge output column/row"))?;
    *cell = value;
    Ok(())
}
impl GraphTable {
    /// Merge complete node/edge tables using checked domain-local identity maps.
    /// Output retains read-only source captures; no graph publication is implicit.
    pub fn merge<I>(tables: I, opts: MergeOptions) -> GraphResult<Self>
    where
        I: IntoIterator<Item = GraphTable>,
    {
        Self::merge_domains(tables.into_iter().collect(), opts, None)
    }
    fn merge_domains(
        tables: Vec<GraphTable>,
        opts: MergeOptions,
        names: Option<Vec<String>>,
    ) -> GraphResult<Self> {
        let policy = merged_policy(&tables)?;
        if let Some(names) = &names {
            if names.len() != tables.len()
                || names.iter().any(|n| n.is_empty())
                || names.iter().collect::<HashSet<_>>().len() != names.len()
            {
                return Err(invalid(
                    "Federated domain names must be nonempty, unique and match bundle count",
                ));
            }
        }
        let domains = tables
            .iter()
            .map(inspect)
            .collect::<GraphResult<Vec<_>>>()?;
        let node_maps = mappings(&domains, true, &opts.strategy)?;
        let edge_maps = mappings(&domains, false, &opts.strategy)?;
        let nodes = merged_component(
            &tables,
            &domains,
            &node_maps,
            &edge_maps,
            true,
            &opts.strategy,
            names.as_deref(),
        )?;
        let edges = merged_component(
            &tables,
            &domains,
            &node_maps,
            &edge_maps,
            false,
            &opts.strategy,
            names.as_deref(),
        )?;
        let mut external = HashSet::new();
        for table in [&nodes, &edges] {
            if let Some(column) = table.column("external_id") {
                for value in column.iter() {
                    if matches!(value, V::Null) {
                        continue;
                    }
                    let id = value
                        .get_text()
                        .ok_or_else(|| invalid("external_id must be text or null"))?;
                    if !external.insert(id) {
                        return Err(invalid("Duplicate external_id in merged graph table; resolve identity explicitly"));
                    }
                }
            }
        }
        let result = Self::with_policy(
            NodesTable::from_base_table(nodes)?,
            EdgesTable::from_base_table(edges)?,
            policy,
        );
        let report = result.validate();
        if !report.is_valid() {
            return Err(invalid(format!(
                "Merged graph table violates validation policy: {}",
                report.errors.join("; ")
            )));
        }
        Ok(result)
    }
    /// Load domain bundles and merge with validated caller-supplied domain names.
    /// Bundle format limitations remain those of load_bundle; no source pool is inferred.
    pub fn from_federated_bundles(
        bundle_paths: Vec<&Path>,
        domain_names: Option<Vec<String>>,
    ) -> GraphResult<Self> {
        if let Some(names) = &domain_names {
            if names.len() != bundle_paths.len() {
                return Err(invalid("Federated domain names must match bundle count"));
            }
        }
        let tables = bundle_paths
            .into_iter()
            .map(Self::load_bundle)
            .collect::<GraphResult<Vec<_>>>()?;
        Self::merge_domains(tables, MergeOptions::domain_prefix(), domain_names)
    }
}
