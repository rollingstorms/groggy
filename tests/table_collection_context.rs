use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table, TableArray},
    },
    AttrValue as V, Graph,
};
use std::{cell::Cell, collections::HashMap};
fn fixture() -> (Graph, BaseTable) {
    let mut g = Graph::new_publishing();
    g.insert_nodes(3).unwrap();
    g.set_node_attrs(HashMap::from([
        (
            "k".into(),
            vec![(0, V::Int(2)), (1, V::Int(1)), (2, V::Int(2))],
        ),
        (
            "x".into(),
            vec![(0, V::Int(10)), (1, V::Int(11)), (2, V::Int(12))],
        ),
        (
            "text".into(),
            vec![
                (0, V::Text("a".into())),
                (1, V::Text("b".into())),
                (2, V::Text("c".into())),
            ],
        ),
    ]))
    .unwrap();
    let t = g.read_view().unwrap().nodes_table().unwrap().into_base();
    (g, t)
}
fn col(t: &BaseTable, name: &str) -> Vec<V> {
    t.column(name).unwrap().data().clone()
}
#[test]
fn empty_selections_retain_schema_roles_sources_and_original_snapshot() {
    let (mut g, t) = fixture();
    let groups = t.groupby(&["k"]).unwrap();
    for empty in [
        groups.take(0),
        groups.skip(100),
        groups.filter(|_| false),
        groups.select_tables(&[]).unwrap(),
    ] {
        assert_eq!(empty.len(), 0);
        let flat = empty.concat().unwrap();
        assert_eq!(flat.column_names(), t.column_names());
        assert_eq!(flat.source_views().len(), 1);
        assert_eq!(flat.source_entity_ids(), None);
        let sum = empty.sum().unwrap();
        assert_eq!(sum.nrows(), 0);
        assert!(sum.has_column("k"));
        assert!(sum.has_column("x"));
        assert!(!sum.has_column("text"));
        assert_eq!(sum.source_views().len(), 1);
        assert!(empty
            .agg(HashMap::from([("missing".into(), "sum".into())]))
            .is_err());
    }
    let empty = groups.take(0);
    g.set_node_attr(0, "x".into(), V::Int(99)).unwrap();
    drop(g);
    assert_eq!(
        empty.concat().unwrap().source_views()[0]
            .node_attr(0, &"x".into())
            .unwrap(),
        Some(V::Int(10))
    );
}
#[test]
fn empty_grouping_retains_declared_keys_and_capture() {
    let (_, t) = fixture();
    let empty = t.head(0).groupby(&["k"]).unwrap();
    let sum = empty.sum().unwrap();
    assert_eq!(sum.nrows(), 0);
    assert!(sum.has_column("k"));
    assert!(sum.has_column("x"));
    assert_eq!(sum.source_views().len(), 1);
    assert!(empty
        .agg(HashMap::from([("k".into(), "sum".into())]))
        .is_err());
    let declared = TableArray::from_group_columns(vec![], &["declared".into()]).unwrap();
    assert!(declared.count().unwrap().has_column("declared"));
}
#[test]
fn filter_evaluates_once_and_invalid_metadata_cannot_be_laundered_through_empty_selection() {
    let (_, t) = fixture();
    let groups = t.groupby(&["k"]).unwrap();
    let calls = Cell::new(0);
    let filtered = groups.filter(|_| {
        calls.set(calls.get() + 1);
        calls.get() == 1
    });
    assert_eq!(calls.get(), 2);
    assert_eq!(filtered.len(), 1);
    assert_eq!(col(&filtered.sum().unwrap(), "k"), vec![V::Int(2)]);
    let mut invalid = groups;
    invalid.group_keys.as_mut().unwrap().pop();
    assert!(invalid.take(0).sum().is_err());
    assert!(invalid.filter(|_| false).concat().is_err());
    assert!(invalid.select_tables(&[]).is_err());
}
#[test]
fn projection_drop_rename_and_empty_controls_keep_a_coherent_schema() {
    let (_, t) = fixture();
    let groups = t.groupby(&["k"]).unwrap();
    let empty = groups
        .take(0)
        .select(&["x".into()])
        .unwrap()
        .rename(&HashMap::from([
            ("x".into(), "score".into()),
            ("k".into(), "kind".into()),
        ]))
        .unwrap();
    let sum = empty.sum().unwrap();
    assert_eq!(sum.column_names(), &["kind", "score"]);
    assert_eq!(sum.source_views().len(), 1);
    assert_eq!(empty.concat().unwrap().column_names(), &["score"]);
    assert!(empty.select(&["missing".into()]).is_err());
    assert!(empty.select(&["score".into(), "score".into()]).is_err());
    assert!(empty.sort_by("missing", true).is_err());
    assert!(empty
        .rename(&HashMap::from([("score".into(), "kind".into())]))
        .is_err());
    assert!(groups
        .rename(&HashMap::from([("x".into(), "k".into())]))
        .is_err());
    let renamed = groups
        .rename(&HashMap::from([("k".into(), "kind".into())]))
        .unwrap();
    assert_eq!(
        col(&renamed.sum().unwrap(), "kind"),
        vec![V::Int(2), V::Int(1)]
    );
    assert_eq!(
        renamed
            .take(0)
            .drop_columns(&["text".into()])
            .unwrap()
            .concat()
            .unwrap()
            .nrows(),
        0
    );
}
#[test]
fn row_empty_and_zero_column_outputs_keep_group_count_and_source_ownership() {
    let (_, t) = fixture();
    let groups = t.groupby(&["k"]).unwrap();
    let head = groups.head(0);
    let sum = head.sum().unwrap();
    assert_eq!(sum.nrows(), 2);
    assert_eq!(col(&sum, "x"), vec![V::Null, V::Null]);
    assert!(!sum.has_column("text"));
    let zero = groups.select(&[]).unwrap().take(0);
    assert!(zero.concat().unwrap().column_names().is_empty());
    assert_eq!(zero.count().unwrap().column_names(), &["k"]);
    assert_eq!(zero.count().unwrap().source_views().len(), 1);
    let retained = groups.select_tables(&[1, 0]).unwrap();
    assert_eq!(
        col(&retained.sum().unwrap(), "k"),
        vec![V::Int(1), V::Int(2)]
    );
    assert_eq!(retained.tables[0].source_entity_ids(), Some(&[1][..]));
}
#[test]
fn callbacks_do_not_run_on_synthetic_empty_inputs_and_source_context_survives_maps() {
    let (_, t) = fixture();
    let empty = t.groupby(&["k"]).unwrap().take(0);
    let calls = Cell::new(0);
    let mapped = empty
        .map(|_| {
            calls.set(calls.get() + 1);
            Ok(BaseTable::new())
        })
        .unwrap();
    assert_eq!(calls.get(), 0);
    assert_eq!(mapped.concat().unwrap().source_views().len(), 1);
    let renamed_empty = t
        .groupby(&["k"])
        .unwrap()
        .head(0)
        .map(|g| g.rename(HashMap::from([("x".into(), "new_score".into())])))
        .unwrap()
        .take(0);
    assert!(renamed_empty.sum().unwrap().has_column("new_score"));
    assert!(!renamed_empty.sum().unwrap().has_column("text"));
    let imported = t
        .groupby(&["k"])
        .unwrap()
        .map(|_| {
            BaseTable::from_columns(HashMap::from([(
                "new".into(),
                BaseArray::from_attr_values(vec![V::Int(1)]),
            )]))
        })
        .unwrap();
    assert_eq!(imported.sum().unwrap().source_views().len(), 1);
    assert_eq!(imported.take(0).concat().unwrap().column_names(), &["new"]);
}
#[test]
fn empty_pairwise_joins_retain_both_sources_and_use_the_real_join_schema() {
    let (_, a) = fixture();
    let (_, b) = fixture();
    let left = a.groupby(&["k"]).unwrap().take(0);
    let right = b.groupby(&["k"]).unwrap().take(0);
    for how in ["inner", "left", "right", "outer", "cross"] {
        let join = left.join(&right, "node_id", how).unwrap();
        let flat = join.concat().unwrap();
        assert_eq!(flat.nrows(), 0);
        assert_eq!(flat.source_views().len(), 2);
        assert!(join
            .count()
            .unwrap()
            .has_column(if matches!(how, "outer" | "cross") {
                "k_right"
            } else {
                "right_k"
            }));
        assert_eq!(join.sum().unwrap().source_views().len(), 2);
    }
    assert!(left.join(&right, "missing", "inner").is_err());
}

#[test]
fn legacy_adapter_source_schema_is_empty_checked_and_retained() {
    let (_, t) = fixture();
    assert!(TableArray::new().with_empty_source(&t).is_err());
    let empty = TableArray::from_group_columns(vec![], &["k".into()])
        .unwrap()
        .with_empty_source(&t.head(0))
        .unwrap();
    assert_eq!(empty.sum().unwrap().source_views().len(), 1);
    assert!(empty.sum().unwrap().has_column("k"));
    assert!(TableArray::from_group_columns(vec![], &["missing".into()])
        .unwrap()
        .with_empty_source(&t.head(0))
        .is_err());
}
