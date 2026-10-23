"""Scalar table predicates keep wide values and exact numeric comparison."""
import groggy as gr


def test_query_comparison_does_not_round_f64_or_large_integers():
    values = [gr.AttrValue.float64(1.0), gr.AttrValue.float64(1 + 2**-40),
              gr.AttrValue.float64(1e300)]
    table = gr.BaseTable.from_dict({"x": values})
    assert table.query("x > 1.0").column("x").to_list() == [1 + 2**-40, 1e300]
    assert table.query("x <= 1.0").column("x").to_list() == [1.0]
    assert table.query("x == 1.0").column("x").to_list() == [1.0]
    table = gr.BaseTable.from_dict({"x": [2**53, 2**53+1, 2**63-1]})
    assert table.query("x > 9007199254740992.0").column("x").to_list() == [2**53+1, 2**63-1]


def test_float_predicate_uses_ieee_nan_and_infinity_rules():
    table = gr.BaseTable.from_dict({"x": [gr.AttrValue.float64(float("nan")),
                                         gr.AttrValue.float64(float("inf")),
                                         gr.AttrValue.float64(0.0)]})
    assert table.query("x == inf").column("x").to_list() == [float("inf")]
    assert len(table.query("x != 0").column("x").to_list()) == 2
