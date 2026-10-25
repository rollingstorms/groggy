use groggy::{
    storage::{
        array::BaseArray,
        table::{EdgesTable, GraphTable, NodesTable},
    },
    viz::{display::DataType, streaming::DataSource},
    AttrValue as V,
};
use std::collections::HashMap;

fn fixture() -> GraphTable {
    GraphTable::new(
        NodesTable::new(vec![7, 30])
            .with_attributes(
                "value".into(),
                HashMap::from([(7, V::Float64(1.23456789012345)), (30, V::Null)]),
            )
            .unwrap(),
        EdgesTable::new(vec![(91, 7, 30)])
            .with_attributes("value".into(), HashMap::from([(91, V::Bool(true))]))
            .unwrap(),
    )
}

#[test]
fn tagged_projection_has_matching_headers_schema_width_and_component_values() {
    let table = fixture();
    let window = table.get_window(0, usize::MAX);
    assert_eq!(window.total_rows, 3);
    assert_eq!(window.rows.len(), 3);
    assert_eq!(
        window.headers,
        vec![
            "entity_kind",
            "node_node_id",
            "node_value",
            "edge_edge_id",
            "edge_source",
            "edge_target",
            "edge_value"
        ]
    );
    assert_eq!(table.total_cols(), 7);
    assert_eq!(window.schema.columns.len(), 7);
    assert!(window.rows.iter().all(|row| row.len() == 7));
    assert_eq!(
        window.rows[0],
        vec![
            V::Text("node".into()),
            V::Int(7),
            V::Float64(1.23456789012345),
            V::Null,
            V::Null,
            V::Null,
            V::Null
        ]
    );
    assert_eq!(
        window.rows[2],
        vec![
            V::Text("edge".into()),
            V::Null,
            V::Null,
            V::Int(91),
            V::Int(7),
            V::Int(30),
            V::Bool(true)
        ]
    );
    assert_eq!(window.schema.columns[2].data_type, DataType::Float);
    assert_eq!(window.schema.columns[6].data_type, DataType::Boolean);
    assert_eq!(table.shape().nodes, (2, 2));
    assert_eq!(table.shape().edges, (1, 4));
}

#[test]
fn windows_cross_the_component_boundary_and_clamp_empty_overflowing_requests() {
    let table = fixture();
    let all = table.get_window(0, 10);
    let middle = table.get_window(1, 2);
    assert_eq!(middle.start_offset, 1);
    assert_eq!(middle.total_rows, 3);
    assert_eq!(middle.rows, all.rows[1..]);
    assert_eq!(middle.headers, all.headers);
    assert_eq!(table.get_window(2, usize::MAX).rows, all.rows[2..]);
    for (start, count) in [(0, 0), (3, 1), (usize::MAX, usize::MAX)] {
        let empty = table.get_window(start, count);
        assert!(empty.rows.is_empty());
        assert_eq!(empty.start_offset, start);
        assert_eq!(empty.headers, all.headers);
        assert_eq!(empty.schema.columns.len(), table.total_cols());
    }
    let empty = GraphTable::empty().get_window(usize::MAX, usize::MAX);
    assert!(empty.rows.is_empty());
    assert_eq!(
        empty.headers,
        vec![
            "entity_kind",
            "node_node_id",
            "edge_edge_id",
            "edge_source",
            "edge_target"
        ]
    );
}

#[test]
fn presentation_identity_is_shared_by_clones_and_changes_before_mutable_access() {
    let original = fixture();
    let mut copy = original.clone();
    let key = original.get_cache_key(0, 2);
    assert_eq!(copy.get_cache_key(0, 2), key);
    assert_ne!(fixture().get_cache_key(0, 2), key);
    copy.nodes_mut()
        .base_table_mut()
        .set_column("value", vec![V::Int(4), V::Int(5)])
        .unwrap();
    assert_ne!(copy.get_cache_key(0, 2), key);
    assert_eq!(original.get_cache_key(0, 2), key);
    let node_key = copy.get_cache_key(0, 2);
    copy.edges_mut()
        .base_table_mut()
        .set_column("value", vec![V::Bool(false)])
        .unwrap();
    assert_ne!(copy.get_cache_key(0, 2), node_key);
    let policy_key = copy.get_cache_key(0, 2);
    copy.set_policy(copy.policy().clone());
    assert_ne!(copy.get_cache_key(0, 2), policy_key);
    assert_eq!(
        original.get_window(0, 1).rows[0][2],
        V::Float64(1.23456789012345)
    );
}

#[test]
fn derived_branches_and_explicit_edge_id_assignment_cannot_alias_cache_keys() {
    let original = fixture();
    let left = original
        .with_column(
            "extra".into(),
            BaseArray::from_attr_values(vec![V::Int(1); 2]),
        )
        .unwrap();
    let right = original
        .with_column(
            "extra".into(),
            BaseArray::from_attr_values(vec![V::Int(2); 2]),
        )
        .unwrap();
    assert_ne!(left.get_cache_key(0, 2), right.get_cache_key(0, 2));
    assert_ne!(
        original.head(1).unwrap().get_cache_key(0, 2),
        original.get_cache_key(0, 2)
    );
    let assigned = original.clone().auto_assign_edge_ids().unwrap();
    assert_ne!(assigned.get_cache_key(0, 2), original.get_cache_key(0, 2));
    assert_eq!(original.to_nodes().node_ids().unwrap(), vec![7, 30]);
}
