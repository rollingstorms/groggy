use groggy::{
    storage::{
        array::{BaseArray, QueryEvaluator},
        table::{BaseTable, TableArray},
    },
    AttrValue as V,
};
use std::collections::HashMap;
fn table(value: V) -> BaseTable {
    BaseTable::from_columns(HashMap::from([("x".into(), BaseArray::new(vec![value]))])).unwrap()
}
#[test]
fn array_and_table_predicates_use_exact_wide_comparisons_for_every_operator() {
    let cases = [
        (
            V::Int(9_007_199_254_740_993),
            "9007199254740992.0",
            [false, true, false, false, true, true],
        ),
        (
            V::Float64(1. + f64::EPSILON),
            "1.0",
            [false, true, false, false, true, true],
        ),
        (
            V::Float64(1. + f64::EPSILON),
            "1.0000000000000002",
            [true, false, false, true, false, true],
        ),
        (
            V::SmallInt(2),
            "2.0",
            [true, false, false, true, false, true],
        ),
        (
            V::Float64(f64::NAN),
            "0",
            [false, true, false, false, false, false],
        ),
        (
            V::Float64(f64::INFINITY),
            "inf",
            [true, false, false, true, false, true],
        ),
        (
            V::Float(1.5),
            "1.5",
            [true, false, false, true, false, true],
        ),
        (
            V::Float64(1e300),
            "1e299",
            [false, true, false, false, true, true],
        ),
    ];
    for (value, literal, expected) in cases {
        for (operator, want) in ["==", "!=", "<", "<=", ">", ">="].into_iter().zip(expected) {
            let evaluator = QueryEvaluator::new(&format!("value {operator} {literal}"));
            assert_eq!(
                evaluator.evaluate(&value).unwrap(),
                want,
                "array {value:?} {operator} {literal}"
            );
            assert_eq!(
                BaseTable::compare_attr_value(&value, operator, literal).unwrap(),
                want,
                "table {value:?} {operator} {literal}"
            );
        }
    }
}
#[test]
fn length_parity_and_cross_table_extrema_do_not_narrow_values() {
    assert!(QueryEvaluator::new("length > 2")
        .evaluate(&V::Float64Vec(vec![0.; 3]))
        .unwrap());
    let even = QueryEvaluator::new("value % 2 == 0");
    for value in [
        V::Float64(4.5),
        V::Float64(f64::NAN),
        V::Float64(f64::INFINITY),
        V::Float(4.5),
    ] {
        assert!(!even.evaluate(&value).unwrap());
    }
    assert!(even.evaluate(&V::Float64(-2.)).unwrap());
    let tables = TableArray::from_tables(vec![
        table(V::Int(9_007_199_254_740_993)),
        table(V::Float64(9_007_199_254_740_992.)),
    ]);
    assert_eq!(
        tables.min_column("x").unwrap(),
        V::Float64(9_007_199_254_740_992.)
    );
    assert_eq!(
        tables.max_column("x").unwrap(),
        V::Int(9_007_199_254_740_993)
    );
}
