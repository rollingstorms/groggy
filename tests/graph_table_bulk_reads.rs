use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable},
    },
    AttrValue as V,
};
use std::collections::{HashMap, HashSet};
fn nodes(ids: Vec<V>, values: Vec<V>) -> NodesTable {
    NodesTable::from_base_table(
        BaseTable::with_column_order(
            HashMap::from([
                ("node_id".into(), BaseArray::from_attr_values(ids)),
                ("x".into(), BaseArray::from_attr_values(values)),
            ]),
            vec!["node_id".into(), "x".into()],
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn bulk_identity_lookup_preserves_first_whole_rows_unknown_ids_and_duplicates() {
    let table = GraphTable::new(
        nodes(
            vec![V::SmallInt(7), V::Int(3), V::Int(7)],
            vec![V::Int(10), V::Null, V::Int(99)],
        ),
        EdgesTable::new(vec![(9, 7, 3), (5, 3, 7), (9, 3, 3)]),
    );
    let result = table.get_nodes(&[3, 7, 7, 999]).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[&7]["x"], V::Int(10));
    assert_eq!(result[&3]["x"], V::Null);
    let result = table.get_edges(&[5, 9, 9, 999]).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[&9]["source"], V::Int(7));
    assert!(table.get_nodes(&[]).unwrap().is_empty());
    assert!(table.get_edges(&[]).unwrap().is_empty());
}
#[test]
fn neighbors_matches_an_independent_small_edge_oracle_for_every_selected_subset() {
    let edges = vec![
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 2),
        (4, 1, 2),
        (5, 3, 1),
        (6, 3, 3),
    ];
    let table = GraphTable::new(
        NodesTable::new(vec![0, 1, 2, 3, 4]),
        EdgesTable::new(edges.clone()),
    );
    for mask in 0..32 {
        let selected: HashSet<_> = (0..5).filter(|node| mask & (1 << node) != 0).collect();
        let mut expected = HashSet::new();
        for &(_, source, target) in &edges {
            if selected.contains(&source) {
                expected.insert(target);
            }
            if selected.contains(&target) {
                expected.insert(source);
            }
        }
        expected.retain(|node| !selected.contains(node));
        assert_eq!(
            table
                .neighbors(&selected.iter().copied().collect::<Vec<_>>())
                .unwrap(),
            expected
        );
    }
}
#[test]
fn bulk_reads_reject_malformed_structural_columns_even_for_empty_queries() {
    let table = GraphTable::new(
        nodes(vec![V::Int(-1)], vec![V::Int(1)]),
        EdgesTable::new(vec![]),
    );
    assert!(table.get_nodes(&[]).is_err());
    let mut table = GraphTable::new(NodesTable::new(vec![0]), EdgesTable::new(vec![(0, 0, 0)]));
    table
        .edges_mut()
        .base_table_mut()
        .set_column("source", vec![V::Null])
        .unwrap();
    assert!(table.neighbors(&[]).is_err());
    table
        .edges_mut()
        .base_table_mut()
        .set_column("edge_id", vec![V::Float64(0.0)])
        .unwrap();
    assert!(table.get_edges(&[]).is_err());
}
#[test]
fn wide_ids_and_owned_lookup_values_survive_later_input_edits() {
    let wide = (1usize << 53) + 1;
    let mut table = GraphTable::new(
        NodesTable::new(vec![wide]),
        EdgesTable::new(vec![(wide, wide, wide + 1)]),
    );
    let values = table.get_nodes(&[wide]).unwrap();
    table
        .nodes_mut()
        .base_table_mut()
        .set_column("node_id", vec![V::Int(3)])
        .unwrap();
    assert_eq!(values[&wide]["node_id"], V::Int(wide as i64));
    assert_eq!(table.neighbors(&[wide]).unwrap(), HashSet::from([wide + 1]));
}
