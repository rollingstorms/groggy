use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue as V,
};
fn table(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    BaseTable::from_columns(
        columns
            .into_iter()
            .map(|(k, v)| (k.into(), BaseArray::new(v)))
            .collect(),
    )
    .unwrap()
}
#[test]
fn f64_columns_participate_in_describe_ranking_and_all_column_windows() {
    let x = 1. + 2f64.powi(-40);
    let source = table(vec![
        ("x", vec![V::Float64(x), V::Float64(1.), V::Float64(3.)]),
        (
            "small",
            vec![V::SmallInt(1), V::SmallInt(2), V::SmallInt(3)],
        ),
    ]);
    let summary = source.describe().unwrap();
    assert!(summary.column("x").is_some());
    assert_eq!(
        source.nlargest(2, "x").unwrap().column("x").unwrap().data(),
        &[V::Float64(3.), V::Float64(x)]
    );
    assert_eq!(
        source
            .nsmallest(2, "x")
            .unwrap()
            .column("x")
            .unwrap()
            .data(),
        &[V::Float64(1.), V::Float64(x)]
    );
    let rolling = source.rolling_all(2, "mean").unwrap();
    assert_eq!(
        rolling.column("x_rolling_2_mean").unwrap().data()[1],
        V::Float64((x + 1.) / 2.)
    );
    assert!(rolling.column("small_rolling_2_mean").is_some());
    let expanding = source.expanding_all("sum").unwrap();
    assert_eq!(
        expanding.column("x_expanding_sum").unwrap().data()[0],
        V::Float64(x)
    );
    assert!(expanding.column("small_expanding_sum").is_some());
}
#[test]
fn pairwise_statistics_use_shared_checked_kernels_on_diagonals_too() {
    let source = table(vec![
        ("x", vec![V::Float64(1.), V::Float64(2.), V::Float64(3.)]),
        ("y", vec![V::Float64(2.), V::Float64(4.), V::Float64(6.)]),
    ]);
    let corr = source.corr("pearson").unwrap();
    for column in ["x", "y"] {
        for value in corr.column(column).unwrap().data() {
            assert!(matches!(value,V::Float64(v) if (v-1.).abs()<1e-15));
        }
    }
    assert_eq!(
        source.cov(1).unwrap().column("x").unwrap().data(),
        &[V::Float64(1.), V::Float64(2.)]
    );
    assert!(source.corr("unsupported").is_err());
    assert!(source.cov(-1).is_err());
    let constant = table(vec![("x", vec![V::Float64(1.); 3])]);
    let column = constant.column("x").unwrap();
    assert_eq!(
        constant
            .corr("pearson")
            .unwrap()
            .column("x")
            .unwrap()
            .data()[0],
        column.corr(column, "pearson").unwrap()
    );
    let invalid = table(vec![("x", vec![V::Float64(f64::NAN)])]);
    assert!(invalid.corr("pearson").is_err());
    assert!(invalid.cov(0).is_err());
}
