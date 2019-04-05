import pytest
import numpy as np

from inferpy.util.iterables import get_shape


@pytest.mark.parametrize('x, expected', [
    # an integer
    (1, ()),
    # a float
    (1.0, ()),
    # a string
    ("foo", ()),
    # a numpy number
    (np.int(1), ()),
    # a numpy array single dimension
    (np.ones(3), (3)),
    # a numpy array two dimension
    (np.ones((2, 3)), (2, 3)),
    # a numpy array three dimension
    (np.ones((2, 3, 4)), (2, 3, 4)),
    # a dict
    (dict(x=1), ()),
    # a dict with a list
    (dict(x=[1, 2, 3]), ()),
    # an empty list
    ([], ()),
    # a list
    ([1, 2, 3], (3)),
    # a list in a list
    ([[1, 2], [3, 4], [5, 6]], (3, 2)),
])
def test_iterables(x, expected):
    get_shape(x) == expected


def test_iterables_exception():
    x = [[1, 2], [3]]  # different number of elements per index in 0-dimension
    with pytest.raises(ValueError):
        get_shape(x)
