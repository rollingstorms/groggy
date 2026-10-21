"""Typed native NumArray reduction references, including integer precision."""
from fractions import Fraction
from itertools import permutations
import math
import statistics

import numpy as np
import pytest
from groggy import NumArray


@pytest.mark.parametrize("dtype",["int32","int64","float32","float64","bool"])
def test_typed_reductions_and_empty_singleton_contract(dtype):
    values=[True,False,True,False] if dtype=="bool" else [1,3,5,7]
    array=NumArray(values,dtype=dtype)
    expected=[int(x) for x in values]
    assert array.sum()==sum(expected)
    assert array.mean()==statistics.mean(expected)
    assert array.var()==pytest.approx(statistics.variance(expected))
    assert array.std()==pytest.approx(statistics.stdev(expected))
    assert array.min()==min(expected) and array.max()==max(expected)
    assert array.median()==statistics.median(expected)
    empty=NumArray([],dtype=dtype)
    assert empty.sum()==0.0
    for operation in ["mean","var","std","min","max","median"]: assert getattr(empty,operation)() is None
    single=NumArray(values[:1],dtype=dtype)
    assert single.var()==single.std()==0.0


def test_int64_precision_is_retained_before_float_output_conversion():
    for values in permutations([2**63-1,1,-(2**63-1)]):
        array=NumArray(list(values),dtype="int64")
        assert array.sum()==1.0 and array.mean()==1.0/3.0
    for offset in [2**63-3,-2**63]:
        values=[offset,offset+1,offset+2]
        array=NumArray(values,dtype="int64")
        variance=float(statistics.variance([Fraction(x) for x in values]))
        assert array.var()==variance==1.0 and array.std()==1.0
    assert NumArray([-2**63,2**63-1],dtype="int64").median()==-0.5


@pytest.mark.parametrize("dtype",["float32","float64"])
def test_cancellation_and_large_finite_inputs(dtype):
    array=NumArray([1e20,1.0,-1e20],dtype=dtype)
    assert array.sum()==1.0 and array.mean()==1.0/3.0
    large=float(np.finfo(np.dtype(dtype)).max)
    array=NumArray([large,large],dtype=dtype)
    assert array.mean()==large and array.median()==large and array.var()==0.0
    assert NumArray([-large,large],dtype=dtype).median()==0.0


@pytest.mark.parametrize("scale",[1e-300,np.finfo(np.float64).max])
def test_std_does_not_require_representable_variance(scale):
    array=NumArray([0.0,float(scale)],dtype="float64")
    assert array.std()==pytest.approx(scale/math.sqrt(2.0),rel=1e-14,abs=0.0)


@pytest.mark.parametrize("dtype",["float32","float64"])
@pytest.mark.parametrize("values",[[float("nan")],[0.0,float("nan")],[float("nan"),0.0]])
def test_nan_propagates_for_every_nonempty_reduction(dtype,values):
    array=NumArray(values,dtype=dtype)
    for operation in ["sum","mean","var","std","min","max","median"]:
        assert math.isnan(getattr(array,operation)())


def test_infinities_have_explicit_ieee_behavior():
    array=NumArray([float("inf"),float("inf")],dtype="float64")
    assert array.sum()==array.mean()==array.median()==float("inf")
    assert math.isnan(array.var()) and math.isnan(array.std())
    opposed=NumArray([-float("inf"),float("inf")],dtype="float64")
    assert math.isnan(opposed.sum()) and math.isnan(opposed.mean()) and math.isnan(opposed.median())
