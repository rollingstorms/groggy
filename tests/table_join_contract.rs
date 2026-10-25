use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table, TableArray},
    },
    types::{CompactString, CompressedData},
    AttrValue as V,
};
use std::collections::HashMap;
fn table(columns: &[(&str, Vec<V>)]) -> BaseTable {
    BaseTable::from_columns(
        columns
            .iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v.clone())))
            .collect(),
    )
    .unwrap()
}
fn values(t: &BaseTable, name: &str) -> Vec<V> {
    t.column(name).unwrap().data().clone()
}
#[test]
fn keys_follow_property_identity_across_storage_and_float_encodings() {
    let left = vec![
        V::Int(7),
        V::Text("hi".into()),
        V::Float(-0.0),
        V::Float(f32::NAN),
        V::Float64(-0.0),
        V::Float64(f64::NAN),
        V::FloatVec(vec![1., -0.]),
        V::Null,
    ];
    let right = vec![
        V::SmallInt(7),
        V::CompactText(CompactString::new("hi")),
        V::Float(0.0),
        V::Float(f32::from_bits(0x7fc00001)),
        V::Float64(0.0),
        V::Float64(f64::from_bits(0x7ff8000000000001)),
        V::CompressedFloatVec(CompressedData::compress_float_vec(&[1., 0.])),
        V::Null,
    ];
    let a = table(&[("key", left.clone())]);
    let b = table(&[("key", right.clone())]);
    assert_eq!(a.inner_join(&b, "key", "key").unwrap().nrows(), left.len());
    let compressed = table(&[(
        "key",
        vec![V::CompressedText(CompressedData::compress_text("hi"))],
    )]);
    assert_eq!(a.inner_join(&compressed, "key", "key").unwrap().nrows(), 1);
    let distinct = table(&[(
        "key",
        vec![
            V::Int(1),
            V::Float(1.),
            V::Float64(1.),
            V::Bool(true),
            V::Text("1".into()),
            V::Json("1".into()),
            V::Bytes(vec![1]),
        ],
    )]);
    assert_eq!(
        distinct
            .inner_join(&distinct, "key", "key")
            .unwrap()
            .nrows(),
        7
    );
}
#[test]
fn duplicate_matches_outer_coalescing_and_order_are_stable() {
    let a = table(&[
        ("k", vec![V::Int(2), V::Int(1), V::Int(2), V::Null]),
        ("a", (0..4).map(V::Int).collect()),
    ]);
    let b = table(&[
        ("k", vec![V::Int(2), V::Int(3), V::Int(2), V::Null]),
        ("b", (10..14).map(V::Int).collect()),
    ]);
    let joined = a.outer_join(&b, "k", "k").unwrap();
    assert_eq!(
        values(&joined, "a"),
        vec![
            V::Int(0),
            V::Int(0),
            V::Int(1),
            V::Int(2),
            V::Int(2),
            V::Int(3),
            V::Null
        ]
    );
    assert_eq!(
        values(&joined, "b"),
        vec![
            V::Int(10),
            V::Int(12),
            V::Null,
            V::Int(10),
            V::Int(12),
            V::Int(13),
            V::Int(11)
        ]
    );
    assert_eq!(
        values(&joined, "k"),
        vec![
            V::Int(2),
            V::Int(2),
            V::Int(1),
            V::Int(2),
            V::Int(2),
            V::Null,
            V::Int(3)
        ]
    );
    for _ in 0..8 {
        assert_eq!(
            values(&a.outer_join(&b, "k", "k").unwrap(), "b"),
            values(&joined, "b")
        );
    }
    let right = a.right_join(&b, "k", "k").unwrap();
    assert_eq!(
        values(&right, "b"),
        vec![
            V::Int(10),
            V::Int(10),
            V::Int(11),
            V::Int(12),
            V::Int(12),
            V::Int(13)
        ]
    );
}
#[test]
fn composite_keys_distinguish_components_and_validate_before_joining() {
    let a = table(&[
        ("x", vec![V::Int(1), V::Int(1)]),
        ("y", vec![V::Text("a".into()), V::Text("b".into())]),
    ]);
    let b = table(&[("u", vec![V::Int(1)]), ("v", vec![V::Text("b".into())])]);
    let joined = a.join_on(&b, &["x", "y"], &["u", "v"], "inner").unwrap();
    assert_eq!(joined.nrows(), 1);
    assert_eq!(values(&joined, "y"), vec![V::Text("b".into())]);
    for (l, r) in [
        (vec![], vec![]),
        (vec!["x"], vec!["u", "v"]),
        (vec!["x", "x"], vec!["u", "v"]),
        (vec!["missing"], vec!["u"]),
    ] {
        assert!(a.join_on(&b, &l, &r, "inner").is_err());
    }
    assert!(a.join_on(&b, &["x"], &["u"], "bogus").is_err());
}
#[test]
fn generated_names_never_overwrite_input_columns() {
    let a = table(&[
        ("k", vec![V::Int(1)]),
        ("v", vec![V::Int(2)]),
        ("left_v", vec![V::Int(3)]),
        ("v_right", vec![V::Int(4)]),
    ]);
    let b = table(&[("k", vec![V::Int(1)]), ("v", vec![V::Int(5)])]);
    for result in [
        a.inner_join(&b, "k", "k"),
        a.left_join(&b, "k", "k"),
        a.outer_join(&b, "k", "k"),
        a.cross_join(&b),
    ] {
        assert!(result.unwrap_err().to_string().contains("collides"));
    }
    assert_eq!(values(&a, "v"), vec![V::Int(2)]);
}
#[test]
fn empty_schemas_and_zero_column_cartesian_products_preserve_shape() {
    let a = table(&[("k", vec![V::Int(1), V::Int(2)])]);
    let empty = a.head(0);
    assert_eq!(a.inner_join(&empty, "k", "k").unwrap().nrows(), 0);
    assert_eq!(a.left_join(&empty, "k", "k").unwrap().nrows(), 2);
    let outer = empty.outer_join(&a, "k", "k").unwrap();
    assert_eq!(values(&outer, "k"), values(&a, "k"));
    let zero = a.select(&[]).unwrap();
    assert_eq!(zero.nrows(), 2);
    let cross = zero.cross_join(&zero).unwrap();
    assert_eq!(cross.nrows(), 4);
    assert!(cross.column_names().is_empty());
}
#[test]
fn table_array_all_join_families_delegate_to_the_same_contract() {
    let a = TableArray::from_tables(vec![table(&[("k", vec![V::Int(1)])])]);
    let b = TableArray::from_tables(vec![table(&[("k", vec![V::Int(2)])])]);
    for (how, rows) in [
        ("inner", 0),
        ("left", 1),
        ("right", 1),
        ("outer", 2),
        ("cross", 1),
    ] {
        assert_eq!(a.join(&b, "k", how).unwrap().tables[0].nrows(), rows);
    }
    assert!(TableArray::new()
        .join(&TableArray::new(), "k", "bad")
        .is_err());
    let keys = HashMap::from([("k".into(), V::Int(1))]);
    let grouped = TableArray::from_tables_with_keys(a.tables.clone(), vec![keys]);
    assert!(grouped
        .join(&grouped, "k", "outer")
        .unwrap()
        .group_keys
        .is_some());
}

#[test]
fn all_keyed_families_match_an_independent_nested_loop_oracle() {
    // Exhaustively vary small row patterns, including duplicate and unmatched/null keys.
    for a_len in 0..5 {
        for b_len in 0..5 {
            for offset in 0..3 {
                let keys = [V::Null, V::Int(1), V::Int(2)];
                let ak: Vec<_> = (0..a_len).map(|i| keys[i % 3].clone()).collect();
                let bk: Vec<_> = (0..b_len).map(|i| keys[(i + offset) % 3].clone()).collect();
                let a = table(&[
                    ("k", ak.clone()),
                    ("a", (0..a_len as i64).map(V::Int).collect()),
                ]);
                let b = table(&[
                    ("k", bk.clone()),
                    ("b", (0..b_len as i64).map(V::Int).collect()),
                ]);
                for how in ["inner", "left", "right", "outer"] {
                    let (left, right) = if how == "right" {
                        (&bk, &ak)
                    } else {
                        (&ak, &bk)
                    };
                    let mut expected = Vec::new();
                    let mut matched = vec![false; right.len()];
                    for (i, k) in left.iter().enumerate() {
                        let mut found = false;
                        for (j, r) in right.iter().enumerate() {
                            if k == r {
                                expected.push((Some(i), Some(j)));
                                matched[j] = true;
                                found = true;
                            }
                        }
                        if !found && how != "inner" {
                            expected.push((Some(i), None));
                        }
                    }
                    if how == "outer" {
                        for (j, m) in matched.iter().enumerate() {
                            if !m {
                                expected.push((None, Some(j)));
                            }
                        }
                    }
                    let result = a.join_on(&b, &["k"], &["k"], how).unwrap();
                    let (l, r) = if how == "right" {
                        ("b", "a")
                    } else {
                        ("a", "b")
                    };
                    assert_eq!(
                        values(&result, l),
                        expected
                            .iter()
                            .map(|(i, _)| i.map(|i| V::Int(i as i64)).unwrap_or(V::Null))
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        values(&result, r),
                        expected
                            .iter()
                            .map(|(_, i)| i.map(|i| V::Int(i as i64)).unwrap_or(V::Null))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
