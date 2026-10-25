use groggy::{
    storage::{
        array::BaseArray,
        graph_table::MergeOptions,
        table::{
            BaseTable, ConflictResolution as S, EdgeConfig, EdgesTable, GraphTable, NodesTable,
            Table, ValidationPolicy, ValidationStrictness,
        },
    },
    AttrValue as V, Graph,
};
use std::collections::HashMap;
fn base(cols: &[(&str, Vec<V>)]) -> BaseTable {
    BaseTable::with_column_order(
        cols.iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v.clone())))
            .collect(),
        cols.iter().map(|(n, _)| n.to_string()).collect(),
    )
    .unwrap()
}
fn fixture(offset: i64, value: i64) -> GraphTable {
    GraphTable::new(
        NodesTable::from_base_table(base(&[
            ("node_id", vec![V::Int(offset + 20), V::Int(offset + 10)]),
            ("x", vec![V::Int(value), V::Float64(1.0000000000000002)]),
            (
                "nodes",
                vec![
                    V::NodeArray(vec![(offset + 10) as usize, (offset + 20) as usize]),
                    V::Null,
                ],
            ),
            (
                "edges",
                vec![V::EdgeArray(vec![(offset + 90) as usize]), V::Null],
            ),
        ]))
        .unwrap(),
        EdgesTable::from_base_table(base(&[
            ("edge_id", vec![V::Int(offset + 90)]),
            ("source", vec![V::Int(offset + 10)]),
            ("target", vec![V::Int(offset + 20)]),
            ("weight", vec![V::Float64(value as f64 + 0.125)]),
        ]))
        .unwrap(),
    )
}
fn merge(t: Vec<GraphTable>, s: S) -> Result<GraphTable, groggy::GraphError> {
    GraphTable::merge(t, MergeOptions::new(s))
}
fn col(t: &BaseTable, n: &str) -> Vec<V> {
    t.column(n).unwrap().data().clone()
}
#[test]
fn domain_prefix_remaps_overlapping_domains_properties_and_typed_references() {
    let a = fixture(0, 1);
    let b = fixture(0, 2);
    let result = merge(vec![a, b], S::DomainPrefix).unwrap();
    let (n, e) = result.components();
    assert_eq!(
        col(n, "node_id"),
        vec![V::Int(0), V::Int(1), V::Int(2), V::Int(3)]
    );
    assert_eq!(col(e, "source"), vec![V::Int(1), V::Int(3)]);
    assert_eq!(col(e, "target"), vec![V::Int(0), V::Int(2)]);
    assert_eq!(
        col(n, "nodes"),
        vec![
            V::NodeArray(vec![1, 0]),
            V::Null,
            V::NodeArray(vec![3, 2]),
            V::Null
        ]
    );
    assert_eq!(
        col(n, "edges"),
        vec![
            V::EdgeArray(vec![0]),
            V::Null,
            V::EdgeArray(vec![1]),
            V::Null
        ]
    );
    assert_eq!(
        col(n, "x"),
        vec![
            V::Int(1),
            V::Float64(1.0000000000000002),
            V::Int(2),
            V::Float64(1.0000000000000002)
        ]
    );
    assert_eq!(col(e, "weight"), vec![V::Float64(1.125), V::Float64(2.125)]);
    assert_eq!(
        col(n, "_original_id"),
        vec![V::Int(20), V::Int(10), V::Int(20), V::Int(10)]
    );
    assert_eq!(
        col(e, "_domain"),
        vec![V::Text("domain_0".into()), V::Text("domain_1".into())]
    );
    assert!(result.validate().is_valid());
    let graph = result.to_graph().unwrap();
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 2);
}
#[test]
fn fail_strategy_preserves_noncolliding_rows_and_rejects_collisions() {
    let result = merge(vec![fixture(0, 1), fixture(100, 2)], S::Fail).unwrap();
    assert_eq!(result.nodes().node_ids().unwrap(), vec![20, 10, 120, 110]);
    assert_eq!(result.edges().edge_ids().unwrap(), vec![90, 190]);
    assert_eq!(col(result.nodes().base(), "x")[2], V::Int(2));
    assert!(merge(vec![fixture(0, 1), fixture(0, 2)], S::Fail).is_err());
}
#[test]
fn auto_remap_preserves_noncollisions_and_reserves_later_input_ids() {
    let result = merge(
        vec![fixture(0, 1), fixture(0, 2), fixture(100, 3)],
        S::AutoRemap,
    )
    .unwrap();
    let ids = result.nodes().node_ids().unwrap();
    assert_eq!(ids, vec![20, 10, 0, 1, 120, 110]);
    assert_eq!(result.edges().edge_ids().unwrap(), vec![90, 0, 190]);
    assert_eq!(
        col(result.nodes().base(), "nodes")[2],
        V::NodeArray(vec![1, 0])
    );
    let small = GraphTable::new(NodesTable::new(vec![0, 1]), EdgesTable::new(vec![]));
    let remapped = merge(vec![fixture(0, 1), fixture(0, 2), small], S::AutoRemap).unwrap();
    assert_eq!(
        remapped.nodes().node_ids().unwrap(),
        vec![20, 10, 2, 3, 0, 1]
    );
}
#[test]
fn keep_policies_choose_whole_rows_while_merge_attributes_preserves_absent_columns() {
    let a = fixture(0, 1);
    let mut b = fixture(0, 2);
    *b.nodes_mut() = NodesTable::from_base_table(
        b.nodes()
            .base()
            .drop_columns(&["edges".into()])
            .unwrap()
            .with_column(
                "later".into(),
                BaseArray::from_attr_values(vec![V::Null, V::Bool(true)]),
            )
            .unwrap(),
    )
    .unwrap();
    let first = merge(vec![a.clone(), b.clone()], S::KeepFirst).unwrap();
    let second = merge(vec![a.clone(), b.clone()], S::KeepSecond).unwrap();
    let attrs = merge(vec![a, b], S::MergeAttributes).unwrap();
    assert_eq!(col(first.nodes().base(), "x")[0], V::Int(1));
    assert_eq!(col(second.nodes().base(), "x")[0], V::Int(2));
    assert_eq!(col(first.nodes().base(), "later"), vec![V::Null, V::Null]);
    assert_eq!(col(second.nodes().base(), "edges"), vec![V::Null, V::Null]);
    assert_eq!(
        col(attrs.nodes().base(), "edges")[0],
        V::EdgeArray(vec![90])
    );
    assert_eq!(col(attrs.nodes().base(), "later")[1], V::Bool(true));
    assert_eq!(col(first.edges().base(), "weight"), vec![V::Float64(1.125)]);
    assert_eq!(
        col(second.edges().base(), "weight"),
        vec![V::Float64(2.125)]
    );
}
#[test]
fn endpoint_conflicts_require_an_explicit_whole_edge_or_remapping_policy() {
    let a = fixture(0, 1);
    let mut b = fixture(0, 2);
    *b.edges_mut() = EdgesTable::new(vec![(90, 20, 10)]);
    assert!(merge(vec![a.clone(), b.clone()], S::MergeAttributes)
        .unwrap_err()
        .to_string()
        .contains("endpoint conflict"));
    assert_eq!(
        merge(vec![a.clone(), b.clone()], S::KeepFirst)
            .unwrap()
            .edges()
            .sources()
            .unwrap(),
        vec![10]
    );
    assert_eq!(
        merge(vec![a, b], S::KeepSecond)
            .unwrap()
            .edges()
            .sources()
            .unwrap(),
        vec![20]
    );
}
#[test]
fn every_strategy_validates_single_input_ids_endpoints_references_and_reserved_metadata() {
    assert!(merge(vec![], S::Fail).is_err());
    for strategy in [
        S::Fail,
        S::KeepFirst,
        S::KeepSecond,
        S::MergeAttributes,
        S::DomainPrefix,
        S::AutoRemap,
    ] {
        let mut duplicate = fixture(0, 1);
        *duplicate.nodes_mut() = NodesTable::new(vec![10, 10]);
        assert!(merge(vec![duplicate], strategy.clone()).is_err());
        let mut dangling = fixture(0, 1);
        *dangling.edges_mut() = EdgesTable::new(vec![(90, 10, 999)]);
        assert!(merge(vec![dangling], strategy.clone()).is_err());
        for value in [
            V::NodeArray(vec![999]),
            V::EdgeArray(vec![999]),
            V::SubgraphRef(7),
        ] {
            let mut bad = fixture(0, 1);
            *bad.nodes_mut() = NodesTable::from_base_table(
                bad.nodes()
                    .base()
                    .with_column(
                        "ref".into(),
                        BaseArray::from_attr_values(vec![value, V::Null]),
                    )
                    .unwrap(),
            )
            .unwrap();
            assert!(merge(vec![bad], strategy.clone()).is_err());
        }
    }
    let mut reserved = fixture(0, 1);
    *reserved.nodes_mut() = NodesTable::from_base_table(
        reserved
            .nodes()
            .base()
            .with_column(
                "_domain".into(),
                BaseArray::from_attr_values(vec![V::Text("custom".into()); 2]),
            )
            .unwrap(),
    )
    .unwrap();
    assert!(merge(vec![reserved], S::DomainPrefix).is_err());
}
#[test]
fn merged_policies_and_external_id_uniqueness_are_enforced() {
    let a = fixture(0, 1);
    let mut b = fixture(100, 2);
    let policy = ValidationPolicy {
        strictness: ValidationStrictness::Strict,
        edge_config: EdgeConfig {
            allow_self_loops: false,
            ..Default::default()
        },
        ..Default::default()
    };
    b.set_policy(policy);
    let out = merge(vec![a.clone(), b], S::Fail).unwrap();
    assert_eq!(out.policy().strictness, ValidationStrictness::Strict);
    assert!(!out.policy().edge_config.allow_self_loops);
    let mut loop_table =
        GraphTable::new(NodesTable::new(vec![1]), EdgesTable::new(vec![(2, 1, 1)]));
    let p = ValidationPolicy {
        edge_config: EdgeConfig {
            allow_self_loops: false,
            ..Default::default()
        },
        ..Default::default()
    };
    loop_table.set_policy(p);
    assert!(merge(vec![loop_table], S::Fail).is_err());
    let a = GraphTable::new(
        a.nodes()
            .clone()
            .with_attributes(
                "external_id".into(),
                HashMap::from([(10, V::Text("same".into()))]),
            )
            .unwrap(),
        a.edges().clone(),
    );
    let b = GraphTable::new(
        NodesTable::new(vec![100])
            .with_attributes(
                "external_id".into(),
                HashMap::from([(100, V::Text("same".into()))]),
            )
            .unwrap(),
        EdgesTable::new(vec![]),
    );
    assert!(merge(vec![a, b], S::DomainPrefix).is_err());
}
#[test]
fn source_pins_remain_immutable_and_merged_rows_require_explicit_write_mapping() {
    let mut a = Graph::new_publishing();
    a.insert_nodes(2).unwrap();
    a.add_edge(0, 1).unwrap();
    let mut b = Graph::new_publishing();
    b.insert_nodes(1).unwrap();
    let table = merge(
        vec![
            a.read_view().unwrap().table().unwrap(),
            b.read_view().unwrap().table().unwrap(),
        ],
        S::DomainPrefix,
    )
    .unwrap();
    assert_eq!(table.nodes().base().source_views().len(), 2);
    assert_eq!(table.nodes().base().source_entity_ids(), None);
    assert_eq!(table.edges().base().source_entity_ids(), None);
    a.set_node_attr(0, "x".into(), V::Int(9)).unwrap();
    assert_eq!(
        table.nodes().base().source_views()[0]
            .node_attr(0, &"x".into())
            .unwrap(),
        None
    );
    assert!(table
        .nodes()
        .base()
        .update_source_graph(&mut a, &["x".into()], None, None)
        .is_err());
}
#[test]
fn federated_bundle_names_are_applied_and_validated() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    GraphTable::new(NodesTable::new(vec![1]), EdgesTable::new(vec![]))
        .save_bundle(&a)
        .unwrap();
    GraphTable::new(NodesTable::new(vec![1]), EdgesTable::new(vec![]))
        .save_bundle(&b)
        .unwrap();
    let result =
        GraphTable::from_federated_bundles(vec![&a, &b], Some(vec!["alpha".into(), "beta".into()]))
            .unwrap();
    assert_eq!(
        col(result.nodes().base(), "_domain"),
        vec![V::Text("alpha".into()), V::Text("beta".into())]
    );
    assert!(GraphTable::from_federated_bundles(
        vec![&a, &b],
        Some(vec!["same".into(), "same".into()])
    )
    .is_err());
    assert!(GraphTable::from_federated_bundles(vec![&a], Some(vec![])).is_err());
}

#[test]
fn dense_domain_projection_matches_independent_small_topology_oracle() {
    let pairs = [(7, 7), (7, 3), (3, 7), (3, 3)];
    for a in 0..16 {
        for b in 0..16 {
            let mut expected = Vec::new();
            let mut tables = Vec::new();
            for (domain, mask) in [a, b].into_iter().enumerate() {
                let mut edges = Vec::new();
                for (i, &(source, target)) in pairs.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        edges.push((30 + i, source, target));
                        expected.push((
                            domain * 2 + usize::from(source == 3),
                            domain * 2 + usize::from(target == 3),
                        ));
                    }
                }
                if mask & 1 != 0 {
                    edges.push((99, 7, 7));
                    expected.push((domain * 2, domain * 2));
                }
                tables.push(GraphTable::new(
                    NodesTable::new(vec![7, 3]),
                    EdgesTable::new(edges),
                ));
            }
            let out = merge(tables, S::DomainPrefix).unwrap();
            assert_eq!(out.nodes().node_ids().unwrap(), vec![0, 1, 2, 3]);
            assert_eq!(
                out.edges()
                    .sources()
                    .unwrap()
                    .into_iter()
                    .zip(out.edges().targets().unwrap())
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                out.edges().edge_ids().unwrap(),
                (0..expected.len()).collect::<Vec<_>>()
            );
        }
    }
}
