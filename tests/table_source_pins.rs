use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, GraphTable, NodesTable, Table},
    },
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch, GraphConfig,
};
use std::collections::HashMap;
fn fixture() -> Graph {
    let mut graph = Graph::new_publishing();
    graph.insert_nodes(4).unwrap();
    graph.add_edges(&[(0, 1), (0, 1), (1, 1), (2, 3)]).unwrap();
    graph
        .set_node_attrs(HashMap::from([(
            "score".into(),
            (0..4).map(|id| (id, V::Int(id as i64))).collect(),
        )]))
        .unwrap();
    graph
        .set_edge_attrs(HashMap::from([(
            "weight".into(),
            (0..4).map(|id| (id, V::Float64(id as f64 + 0.5))).collect(),
        )]))
        .unwrap();
    graph
}
fn replace(table: &BaseTable, name: &str, values: Vec<V>) -> BaseTable {
    table
        .with_column(name.into(), BaseArray::from_attr_values(values))
        .unwrap()
}
#[test]
fn row_selection_sort_filter_projection_and_empty_slices_keep_exact_identity() {
    let mut graph = fixture();
    let view = graph.read_view().unwrap();
    let table = view.nodes_table().unwrap().into_base();
    assert_eq!(table.source_entity_ids(), Some(&[0, 1, 2, 3][..]));
    let selected = table
        .select_rows(&[3, 1, 2])
        .unwrap()
        .sort_by("score", true)
        .unwrap()
        .filter("score >= 2")
        .unwrap()
        .select(&["score".into()])
        .unwrap();
    assert_eq!(selected.source_entity_ids(), Some(&[2, 3][..]));
    assert_eq!(selected.source_entity_kind(), Some("node"));
    let edited = replace(&selected, "score", vec![V::Float64(2.25), V::Null]);
    let before = graph.current_state_id();
    let result = edited
        .update_source_graph(&mut graph, &["score".into()], Some(10), None)
        .unwrap();
    assert_eq!(result.batch.state_id, before + 1);
    assert_eq!(result.view.node_ids(), vec![2, 3]);
    assert_eq!(
        graph.get_node_attr(2, &"score".into()).unwrap(),
        Some(V::Float64(2.25))
    );
    assert_eq!(
        graph.get_node_attr(3, &"score".into()).unwrap(),
        Some(V::Null)
    );
    assert_eq!(view.node_attr(2, &"score".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(selected.column("score").unwrap().get(0), Some(&V::Int(2)));
    let empty = table.head(0);
    assert_eq!(empty.nrows(), 0);
    assert!(empty.has_column("score"));
    assert_eq!(empty.source_entity_ids(), Some(&[][..]));
    assert_eq!(empty.source_views()[0].write_state_id(), Some(before));
    assert!(edited
        .update_source_graph(&mut graph, &["score".into()], None, None)
        .is_err());
}
#[test]
fn edge_projection_without_structural_columns_preserves_parallel_edge_identity() {
    let mut graph = fixture();
    let view = graph.read_view().unwrap().select_edges(&[1, 2]).unwrap();
    let table = view
        .edges_table()
        .unwrap()
        .into_base()
        .select_rows(&[1, 0])
        .unwrap()
        .select(&["weight".into()])
        .unwrap();
    assert_eq!(table.source_entity_ids(), Some(&[2, 1][..]));
    let table = replace(&table, "weight", vec![V::Float64(9.5), V::Float64(7.5)]);
    let result = table
        .update_source_graph(&mut graph, &["weight".into()], None, None)
        .unwrap();
    assert_eq!(result.view.edge_ids(), vec![1, 2]);
    assert_eq!(
        graph.get_edge_attr(0, &"weight".into()).unwrap(),
        Some(V::Float64(0.5))
    );
    assert_eq!(
        graph.get_edge_attr(1, &"weight".into()).unwrap(),
        Some(V::Float64(7.5))
    );
    assert_eq!(
        graph.get_edge_attr(2, &"weight".into()).unwrap(),
        Some(V::Float64(9.5))
    );
}
#[test]
fn mixed_source_tables_publish_once_and_reject_separate_captures() {
    let mut graph = fixture();
    let frozen = graph.read_view().unwrap();
    let tables = frozen.table().unwrap();
    let nodes = replace(tables.nodes().base(), "score", vec![V::Int(8); 4]);
    let edges = replace(tables.edges().base(), "weight", vec![V::Float64(8.5); 4]);
    let combined = GraphTable::new(
        NodesTable::from_base_table(nodes).unwrap(),
        EdgesTable::from_base_table(edges).unwrap(),
    );
    let before = graph.current_state_id();
    let result = combined
        .update_source_graph(
            &mut graph,
            &["score".into()],
            &["weight".into()],
            Some(-10),
            None,
        )
        .unwrap();
    assert_eq!(result.batch.state_id, before + 1);
    assert_eq!(
        graph
            .publication_metadata(before + 1)
            .unwrap()
            .unwrap()
            .change_times
            .len(),
        8
    );
    assert_eq!(
        frozen.edge_attr(0, &"weight".into()).unwrap(),
        Some(V::Float64(0.5))
    );
    let incompatible = GraphTable::new(
        graph.read_view().unwrap().nodes_table().unwrap(),
        graph.read_view().unwrap().edges_table().unwrap(),
    );
    assert!(incompatible
        .update_source_graph(&mut graph, &["score".into()], &[], None, None)
        .unwrap_err()
        .to_string()
        .contains("exact source capture"));
}
#[test]
fn edited_identity_endpoints_duplicates_and_foreign_owners_reject() {
    for edge in [false, true] {
        let mut graph = fixture();
        let pin = graph.read_view().unwrap();
        let table = if edge {
            pin.edges_table().unwrap().into_base()
        } else {
            pin.nodes_table().unwrap().into_base()
        };
        let property = if edge { "weight" } else { "score" };
        let id = if edge { "edge_id" } else { "node_id" };
        for invalid in [
            replace(&table, id, vec![V::Int(3); 4]),
            table.select_rows(&[0, 0]).unwrap(),
        ] {
            assert!(invalid
                .update_source_graph(&mut graph, &[property.into()], None, None)
                .is_err());
        }
        let mut foreign = fixture();
        assert!(table
            .update_source_graph(&mut foreign, &[property.into()], None, None)
            .is_err());
        assert_eq!(graph.current_state_id(), pin.write_state_id().unwrap());
        assert!(!graph.has_uncommitted_changes());
        if edge {
            let bad = replace(&table, "source", vec![V::Int(3); 4]);
            assert!(bad
                .update_source_graph(&mut graph, &[property.into()], None, None)
                .is_err());
        }
    }
}
#[test]
fn joins_keep_all_source_pins_and_require_explicit_write_identity() {
    let graph = fixture();
    let a = graph
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base();
    let other = fixture();
    let b = other
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base();
    for join in [
        a.inner_join(&b, "node_id", "node_id").unwrap(),
        a.left_join(&b, "node_id", "node_id").unwrap(),
        a.right_join(&b, "node_id", "node_id").unwrap(),
        a.outer_join(&b, "node_id", "node_id").unwrap(),
        a.cross_join(&b).unwrap(),
    ] {
        assert_eq!(join.source_views().len(), 2);
        assert_eq!(join.source_entity_ids(), None);
        assert_eq!(join.source_state_id(), None);
        let mut destination = fixture();
        assert!(join
            .update_source_graph(&mut destination, &["score".into()], None, None)
            .unwrap_err()
            .to_string()
            .contains("explicit identity"));
        assert_eq!(join.head(1).source_views().len(), 2);
    }
    let repeated = a.inner_join(&a, "node_id", "node_id").unwrap();
    assert_eq!(repeated.source_views().len(), 1);
    drop(graph);
    assert_eq!(
        a.source_views()[0].node_attr(0, &"score".into()).unwrap(),
        Some(V::Int(0))
    );
}
#[test]
fn empty_projections_row_bounds_and_non_entity_transforms_are_safe() {
    let graph = fixture();
    let table = graph
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base();
    let empty_columns = table.select(&[]).unwrap();
    assert_eq!(empty_columns.nrows(), 4);
    assert!(empty_columns.select_rows(&[99]).is_err());
    let dropped = table.drop_rows(&[0, 0, 2]).unwrap();
    assert_eq!(dropped.nrows(), 2);
    assert_eq!(dropped.source_entity_ids(), Some(&[1, 3][..]));
    let row = table
        .column_names()
        .iter()
        .map(|name| {
            (
                name.clone(),
                table.column(name).unwrap().get(0).unwrap().clone(),
            )
        })
        .collect();
    let appended = table.append_row(row).unwrap();
    assert_eq!(appended.source_state_id(), None);
    assert_eq!(appended.source_views().len(), 1);
    let aggregates = table
        .aggregate(HashMap::from([("score".into(), "sum".into())]))
        .unwrap();
    assert_eq!(aggregates.source_state_id(), None);
    assert_eq!(aggregates.source_views().len(), 1);
}
#[test]
fn durable_source_writeback_recovers_and_rejection_does_not_append() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tables.ggj");
    let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                node_attrs: HashMap::from([(
                    "v".into(),
                    vec![(N::Added(0), V::Int(1)), (N::Added(1), V::Int(2))],
                )]),
                ..Default::default()
            },
            0,
            "seed".into(),
            "test".into(),
        )
        .unwrap();
    let table = store
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base()
        .head(1);
    let edited = replace(&table, "v", vec![V::Int(9)]);
    let result = edited
        .update_source_durable(&mut store, &["v".into()], Some(10), None)
        .unwrap();
    assert_eq!(result.batch.state_id, 2);
    let bytes = std::fs::metadata(&path).unwrap().len();
    assert!(edited
        .update_source_durable(&mut store, &["v".into()], None, None)
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
    drop(store);
    let mut store = DurableGraph::open(&path).unwrap();
    assert_eq!(
        store
            .graph()
            .unwrap()
            .get_node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(9))
    );
    assert!(edited
        .update_source_durable(&mut store, &["v".into()], None, None)
        .is_err());
}

#[test]
fn row_preserving_masks_groups_column_edits_and_missing_value_filters_keep_identity() {
    let graph = fixture();
    let table = graph
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base();
    assert_eq!(table.tail(2).source_entity_ids(), Some(&[2, 3][..]));
    assert_eq!(
        table
            .filter_by_mask(&[true, false, true, false])
            .unwrap()
            .source_entity_ids(),
        Some(&[0, 2][..])
    );
    assert_eq!(
        table
            .sort_values(vec!["score".into()], vec![false])
            .unwrap()
            .source_entity_ids(),
        Some(&[3, 2, 1, 0][..])
    );
    assert_eq!(
        table
            .rename(HashMap::from([("score".into(), "value".into())]))
            .unwrap()
            .source_entity_ids(),
        Some(&[0, 1, 2, 3][..])
    );
    let missing = replace(
        &table,
        "score",
        vec![V::Null, V::Int(1), V::Null, V::Int(3)],
    );
    assert_eq!(
        missing
            .dropna_subset(&["score"])
            .unwrap()
            .source_entity_ids(),
        Some(&[1, 3][..])
    );
    assert_eq!(
        missing.fillna_all(V::Int(0)).unwrap().source_entity_ids(),
        Some(&[0, 1, 2, 3][..])
    );
    for group in table.group_by(&["score".into()]).unwrap() {
        assert_eq!(group.source_entity_ids().unwrap().len(), 1);
    }
}
#[test]
fn mixed_table_kind_confusion_and_unbound_inputs_cannot_guess_identity() {
    let mut graph = fixture();
    let pin = graph.read_view().unwrap();
    let wrong = pin
        .edges_table()
        .unwrap()
        .into_base()
        .with_column(
            "node_id".into(),
            BaseArray::from_attr_values((0..4).map(V::Int).collect()),
        )
        .unwrap();
    let combined = GraphTable::new(
        NodesTable::from_base_table(wrong).unwrap(),
        pin.edges_table().unwrap(),
    );
    assert!(combined
        .update_source_graph(&mut graph, &[], &["weight".into()], None, None)
        .unwrap_err()
        .to_string()
        .contains("row kinds"));
    let plain = BaseTable::from_columns(HashMap::from([(
        "score".into(),
        BaseArray::from_attr_values(vec![V::Int(9)]),
    )]))
    .unwrap();
    assert!(plain
        .update_source_graph(&mut graph, &["score".into()], None, None)
        .unwrap_err()
        .to_string()
        .contains("no pinned graph source"));
}
#[test]
fn schema_rejection_and_read_only_sources_preserve_owner_state() {
    let mut graph = fixture();
    let table = graph
        .read_view()
        .unwrap()
        .nodes_table()
        .unwrap()
        .into_base();
    let labels = replace(&table, "labels", vec![V::TextVec(vec!["Person".into()]); 4]);
    let schema =
        groggy::GraphSchema::from_json(r#"{"node_labels":{"Person":{"name":{"type":"text"}}}}"#)
            .unwrap();
    let head = graph.current_state_id();
    assert!(labels
        .update_source_graph(&mut graph, &["labels".into()], None, Some(&schema))
        .is_err());
    assert_eq!(graph.current_state_id(), head);
    let historic =
        groggy::GraphView::all(graph.snapshot_at_commit(head).unwrap().read_state().clone())
            .nodes_table()
            .unwrap()
            .into_base();
    assert_eq!(historic.source_state_id(), None);
    assert!(historic
        .update_source_graph(&mut graph, &["score".into()], None, None)
        .is_err());
}
