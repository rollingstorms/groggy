import pytest
from groggy import GraphMatrix


def test_slicing_preserves_empty_shapes_reversal_boolean_masks_and_owned_values():
    a = GraphMatrix.from_data([[0., 3.], [1., 4.], [2., 5.]])
    assert a[:, []].shape == (2, 0)
    assert a[[], :].shape == (0, 3)
    assert a[::-1, ::-1].get(0, 0) == 5.
    assert a[[False, True], :].shape == (1, 3)
    assert a[[False, True], :].get(0, 0) == 3.
    assert a[:, [True, False, True]].shape == (2, 2)
    selected = a[[1, 0, 1], [2, 0]]
    assert selected.shape == (3, 2)
    assert selected.get(1, 1) == 0.
    assert a.reshape(3, 2).get(2, 1) == 5.
    with pytest.raises(IndexError):
        a[:, [3]]
    with pytest.raises(IndexError):
        a[::0, :]
