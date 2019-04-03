import pytest
import tensorflow as tf

import inferpy as inf
from inferpy.util import runtime


@pytest.fixture(params=[True, False])
def is_tf_run_default(request):
    inf.set_tf_run(request.param)
    yield request.param


def test_tf_run_allowed(is_tf_run_default):
    # create a function which directly returns a tensor
    @runtime.tf_run_allowed
    def foo():
        return tf.ones(1)

    if is_tf_run_default:
        # in this case the tensor is run in a session
        assert foo() == [1.]
    else:
        # in this case the function returns a tensor
        assert isinstance(foo(), tf.Tensor)


def test_tf_run_allowed_nested(is_tf_run_default):
    # create a function which directly returns a tensor created by a second function
    # store the result of the nested call function bar
    res_bar = None

    @runtime.tf_run_allowed
    def foo():
        return bar()

    @runtime.tf_run_allowed
    def bar():
        nonlocal res_bar
        res_bar = tf.ones(1)
        return res_bar

    if is_tf_run_default:
        # in this case the tensor is run in a session
        assert foo() == [1.]
    else:
        # in this case the function returns a tensor
        assert isinstance(foo(), tf.Tensor)

    # independently of is_tf_run_default, res_bar is always a tensor
    assert isinstance(res_bar, tf.Tensor)


def test_tf_run_ignored(is_tf_run_default):
    # create a function which directly returns a tensor

    @runtime.tf_run_ignored
    def foo():
        return bar()

    @runtime.tf_run_allowed
    def bar():
        return tf.ones(1)

    # independently of is_tf_run_default, foo always returns a tensor
    assert isinstance(foo(), tf.Tensor)
