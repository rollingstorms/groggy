"""BaseArray reduction references use exact rationals over stored i64/f32 values."""
from fractions import Fraction
from itertools import permutations
import math
import statistics

import numpy as np
import pytest
from groggy import AttrValue, BaseArray


def stored(values):
    return [float(np.float32(value)) if isinstance(value, float) else value for value in values]


def exact(values):
    return [Fraction(value) for value in stored(values) if value is not None]


@pytest.mark.parametrize("values", [
    [2**63 - 1, 1, -(2**63 - 1)],
    [2**63 - 1, 2**63 - 1, -(2**63 - 1)],
    [2**63 - 1, -(float(2**63)), 2],
    [1e20, 1.0, -1e20],
    [2**63 - 1, 1, 0.0],
])
def test_sum_and_mean_keep_cancellation_independent_of_input_order(values):
    for order in permutations(values):
        array = BaseArray(list(order))
        total = sum(exact(order))
        expected_sum = float(np.float32(float(total))) if any(isinstance(x, float) for x in order) else int(total)
        assert array.sum() == expected_sum
        assert array.mean() == float(total / len(order))


@pytest.mark.parametrize("offset", [0, 2**63 - 5, -(2**63) + 4])
def test_sample_statistics_preserve_high_offset_integer_differences(offset):
    for residuals in [[0, 1], [-3, 0, 2], [0, 0, 0], [-3, -1, 0, 1, 2]]:
        values = [offset + value for value in residuals]
        array = BaseArray(values)
        # Python 3.9's direct integer variance also loses this high-offset case.
        # Feed exact Fractions so the reference retains the integer differences.
        variance = float(statistics.variance([Fraction(x) for x in values]))
        assert array.var() == pytest.approx(variance, rel=1e-14)
        assert array.std() == pytest.approx(math.sqrt(variance), rel=1e-14)
        assert array.median() == float(statistics.median([Fraction(x) for x in values]))


@pytest.mark.parametrize("values", [
    [2**63 - 1, float(2**63), 2**63 - 2],
    [1.0, None, 3, 5.0, None, 9],
    [float(np.finfo(np.float32).max), -float(np.finfo(np.float32).max)],
])
def test_mixed_statistics_match_rational_reference(values):
    array = BaseArray(values)
    rational = exact(values)
    assert array.mean() == float(statistics.mean(rational))
    assert array.var() == pytest.approx(float(statistics.variance(rational)), rel=1e-14)
    assert array.std() == pytest.approx(math.sqrt(float(statistics.variance(rational))), rel=1e-14)
    assert array.median() == float(statistics.median(rational))


@pytest.mark.parametrize("values", [[-(2**63), 2**63 - 1], [2**63 - 1, -float(2**63)]])
def test_even_median_retains_small_result_after_large_cancellation(values):
    assert BaseArray(values).median() == -0.5


@pytest.mark.parametrize("values", [[2**63 - 1, 1], [-(2**63), -1], [AttrValue.float64(float(np.finfo(np.float64).max))] * 2])
def test_sum_output_overflow_errors_without_preventing_other_statistics(values):
    array = BaseArray(values)
    with pytest.raises(ValueError, match="range"):
        array.sum()
    assert math.isfinite(array.mean())
    assert math.isfinite(array.var())


@pytest.mark.parametrize("values", [[], [None, None], [1, True], ["x"], [float("nan")], [float("inf")], [-float("inf")]])
@pytest.mark.parametrize("operation", ["sum", "mean", "median", "var", "std"])
def test_unsupported_input_errors_without_nan_sorting_or_silent_coercion(values, operation):
    with pytest.raises(ValueError):
        getattr(BaseArray(values), operation)()


def test_null_policy_sample_denominator_types_and_source_ownership():
    array = BaseArray([3, None, 1.0, 2])
    for operation in ["sum", "mean", "median", "var", "std"]:
        assert math.isfinite(getattr(array, operation)())
    assert [array[i] for i in range(len(array))] == [3, None, 1.0, 2]
    assert type(BaseArray([1, 2]).sum()) is int
    assert type(BaseArray([1, 2.0]).sum()) is float
    one = BaseArray([None, 7, None])
    assert one.sum() == one.mean() == one.median() == 7
    for operation in ["var", "std"]:
        with pytest.raises(ValueError):
            getattr(one, operation)()


def test_median_preserves_negative_zero_without_breaking_mixed_zero_ordering():
    assert math.copysign(1.0, BaseArray([-0.0]).median()) == -1.0
    assert math.copysign(1.0, BaseArray([-0.0, -0.0]).median()) == -1.0
    for values in permutations([0, -0.0, 0.0]):
        assert BaseArray(list(values)).median() == 0.0

@pytest.mark.parametrize("values", [
    [2**63-2,2**63-1,None],[-2**63+1,-2**63,None],
    [2**63-1,float(2**63)], [16777217,16777216.0],
    [None,-0.5,0,0.5], [1.0,1],
])
def test_extrema_match_exact_python_mixed_numeric_order(values):
    for order in permutations(values):
        array=BaseArray(list(order))
        reference=[x for x in stored(order) if x is not None]
        for operation in ["min","max"]:
            expected = min(reference) if operation == "min" else max(reference)
            actual = getattr(array,operation)()
            assert actual == expected
            assert type(actual) is type(expected)


@pytest.mark.parametrize("values", [[],[None],[True],["x"],
    [float("nan")],[0,float("inf")],[-float("inf"),0]])
@pytest.mark.parametrize("operation", ["min","max"])
def test_extrema_reject_invalid_domains_including_singletons(values,operation):
    with pytest.raises(ValueError): getattr(BaseArray(values),operation)()


def test_extrema_keep_first_tied_scalar_and_signed_zero():
    for operation in ["min","max"]:
        result=getattr(BaseArray([-0.0,0,0.0]),operation)()
        assert type(result) is float
        assert math.copysign(1.0,result)==-1.0
