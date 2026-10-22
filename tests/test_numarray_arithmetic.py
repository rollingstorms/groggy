import math
import numpy as np
import pytest
from groggy import NumArray


@pytest.mark.parametrize('dtype',['int32','int64','float32','float64'])
def test_vector_arithmetic_scalar_broadcast_and_true_division(dtype):
    a=NumArray([2,4,6],dtype=dtype);b=NumArray([1,2,3],dtype=dtype)
    for result,expected in [(a+b,[3,6,9]),(a-b,[1,2,3]),(a*b,[2,8,18]),(a+2,[4,6,8]),(2-a,[0,-2,-4]),(a*2,[4,8,12])]:
        assert result.dtype==dtype and result.to_list()==expected
    assert (a/b).dtype=='float64' and (a/b).to_list()==[2.,2.,2.]
    assert (a/4).to_list()==[0.5,1.,1.5]
    assert (12/a).to_list()==[6.,3.,2.]
    assert a.dot(b)==28
    assert a.norm()==pytest.approx(math.sqrt(56))
    assert a.to_list()==[2,4,6]
    with pytest.raises(ValueError):a+NumArray([1],dtype=dtype)
    with pytest.raises(ValueError):a/0


def test_integer_limits_comparison_precision_and_typed_unique():
    high=2**63-1;low=-2**63
    a=NumArray([high,1,-high],dtype='int64');ones=NumArray([1,1,1],dtype='int64')
    assert a.dot(ones)==1
    with pytest.raises(ValueError):a+ones
    assert NumArray([low]).dtype=='int64'
    values=NumArray([high,high-1,high],dtype='int64')
    assert values.unique().dtype=='int64' and values.unique().to_list()==[high,high-1]
    assert (values==high).to_list()==[True,False,True]
    assert (values<float(high)).to_list()==[True,True,True]
    assert (values==float(high)).to_list()==[False,False,False]
    assert (NumArray([low],dtype='int64')>low-1).to_list()==[True]
    with pytest.raises(TypeError):values+NumArray([1,2,3],dtype='float64')


def test_conversion_policy_keeps_rounding_and_rejects_range_errors():
    assert NumArray([2.5,-2.5],dtype='int32').to_list()==[3,-3]
    assert NumArray([2.5,-2.5]).to_type('int64').to_list()==[3,-3]
    for value,dtype in [(2**31,'int32'),(2**63,'int64'),(float(2**63),'int64'),(math.inf,'int32'),(1e300,'float32')]:
        with pytest.raises((ValueError,OverflowError)):NumArray([value],dtype=dtype)
    with pytest.raises(ValueError):NumArray([2**63-1],dtype='int64').to_type('float64',exact=True)
    with pytest.raises(ValueError):NumArray([1.5]).to_type('int64',exact=True)
    with pytest.raises(ValueError):NumArray([2**63-1],dtype='int64').reshape(1,1)
    assert NumArray([1,2],dtype='int64').reshape(1,2).shape==(1,2)


def test_ieee_comparisons_membership_unique_and_float_ranges():
    a=NumArray([math.inf,1.,math.nan])
    assert (a==math.inf).to_list()==[True,False,False]
    assert (a!=math.inf).to_list()==[False,True,True]
    assert a.contains(math.inf)
    assert not a.contains(1.+np.finfo(float).eps)
    values=NumArray([math.nan,math.nan,-0.,0.,2.]).unique().to_list()
    assert len(values)==3 and math.isnan(values[0]) and math.copysign(1,values[1])<0
    assert NumArray([1e308,1e308]).norm()==pytest.approx(math.hypot(1e308,1e308))
    assert NumArray([1e308,1.,-1e308]).dot(NumArray([1.,1.,1.]))==1.
