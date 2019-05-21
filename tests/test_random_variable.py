import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import pytest
from tests import no_raised_exc

import inferpy as inf
from inferpy.models import random_variable


# TODO: boolean operators and iter are not tested
@pytest.mark.parametrize("tensor, expected", [
    # sum
    (inf.Normal([0., 1.], 0) + inf.Normal([1., 2.], 0), [1., 3.]),
    ([0, 1] + inf.Normal([1., 2.], 0), [1., 3.]),
    (inf.Normal([0., 1.], 0) + [1, 2], [1., 3.]),
    # sub
    (inf.Normal([0., 1.], 0) - inf.Normal([1., 2.], 0), [-1., -1.]),
    ([0, 1] - inf.Normal([1., 2.], 0), [-1., -1.]),
    (inf.Normal([0., 1.], 0) - [1, 2], [-1., -1.]),
    # mult
    (inf.Normal([0., 1.], 0) * inf.Normal([1., 2.], 0), [0., 2.]),
    ([0, 1] * inf.Normal([1., 2.], 0), [0., 2.]),
    (inf.Normal([0., 1.], 0) * [1, 2], [0., 2.]),
    # div
    (inf.Normal([0., 1.], 0) / inf.Normal([1., 2.], 0), [0., 0.5]),
    ([0, 1] / inf.Normal([1., 2.], 0), [0., 0.5]),
    (inf.Normal([0., 1.], 0) / [1, 2], [0., 0.5]),
    # mod
    (inf.Normal([0., 1.], 0) % inf.Normal([2., 2.], 0), [0., 1.]),
    ([0, 1] % inf.Normal([2., 2.], 0), [0., 1.]),
    (inf.Normal([0., 1.], 0) % [2, 2], [0., 1.]),
    # lt
    (inf.Normal(0., 0) < inf.Normal(2., 0), True),
    (1 < inf.Normal(0., 0), False),
    (inf.Normal(1., 0) < 2, True),
    # le
    (inf.Normal(2., 0) <= inf.Normal(2., 0), True),
    (1 <= inf.Normal(0., 0), False),
    (inf.Normal(1., 0) <= 1, True),
    # gt
    (inf.Normal(5., 0) > inf.Normal(2., 0), True),
    (1 > inf.Normal(3., 0), False),
    (inf.Normal(3., 0) > 1, True),
    # getitem
    (inf.Normal([0., 1., 2., 3.], 0)[0], 0),
    (inf.Normal([0., 1., 2., 3.], 0)[2], 2),
    (inf.Normal(tf.ones((3, 2)), 0)[2][0], 1),
    # pow
    (inf.Normal(2., 0) ** inf.Normal(3., 0), 8.),
    (3 ** inf.Normal(2., 0), 9.),
    (inf.Normal(3., 0) ** 3, 27.),
    # neg
    (-inf.Normal(5., 0), -5),
    (-inf.Normal(-5., 0), 5),
    # abs
    (abs(inf.Normal(5., 0)), 5),
    (abs(inf.Normal(-5., 0)), 5),
    # matmul
    (tf.matmul(inf.Normal(tf.ones((2, 3), dtype=np.float32), 0),
               inf.Normal(tf.eye(3, dtype=np.float32), 0)), np.ones((2, 3))),
    (tf.matmul(np.ones((2, 3), dtype=np.float32),
               inf.Normal(tf.eye(3, dtype=np.float32), 0)), np.ones((2, 3))),
    (tf.matmul(inf.Normal(np.ones((2, 3), dtype=np.float32), 0),
               np.eye(3, dtype=np.float32)), np.ones((2, 3))),
])
def test_operations(tensor, expected):
    with tensor.graph.as_default():
        # evaluate the tensor from the operation
        with tf.Session() as sess:
            result = sess.run(tensor)
        # assert that it is equal to expected
        assert np.array_equal(result, expected)


@pytest.mark.parametrize("model_object", [
    # Simple random variable using scalars as parameters
    (inf.Normal(0, 1)),
    # Simple random variable using a list as parameter
    (inf.Normal([0., 0., 0., 0.], 1)),
    # Simple random variable using a numpy array as parameter
    (inf.Normal(np.zeros(5), 1)),
    # Simple random variable using a tensor as parameter
    (inf.Normal(0, tf.ones(5))),
    # Simple random variable using another random variable as parameter
    (inf.Normal(inf.Normal(0, 1), 1)),
    # Simple random variable using a combination of the previously tested options as parameter
    (inf.Normal([inf.Normal(0, 1), 1., tf.constant(1.)], 1.)),
    # Random variable operation used to define a Random Variable
    (inf.Normal(inf.Normal(0, 1) + inf.Normal(0, 1), 1)),
])
def test_edward_type(model_object):
    assert isinstance(model_object.var, ed.RandomVariable)


def test_name():
    x = inf.Normal(0, 1, name='foo')
    assert x.name == 'foo'

    # using the name, not the tensor name
    x = inf.Normal(0, 1, name='foo')
    assert x.name == 'foo'

    # Automatic name generation. It starts with 'randvar_X', where initially X is 0
    x = inf.Normal(0, 1)
    assert isinstance(x.name, str)
    assert x.name == 'randvar_0'


def test_tensor_register():
    # This allows to run a inferpy.inf.RandomVariable directly in a tf session.

    x = inf.Normal(5., 0., name='foo')

    assert inf.get_session().run(x) == 5.
    assert isinstance(tf.convert_to_tensor(x), tf.Tensor)
    assert inf.get_session().run(tf.convert_to_tensor(x)) == 5.
    assert inf.get_session().run(tf.constant(5.) + x) == 10.
    assert inf.get_session().run(x + tf.constant(5.)) == 10.


@pytest.mark.parametrize("arg, bc_shape, expected_flow, expected_result", [
    # No bc_shape makes no broadcast
    (1, None, no_raised_exc(), 1),
    # Element not list and without shape makes no broadcast
    (1, (2, 3), no_raised_exc(), 1),
    # list and bc_shape, bc can be done
    ([1], (2,), no_raised_exc(), [1, 1]),
    # list and bc_shape, bc can be done but shape is the same
    ([1., 2.], (2,), no_raised_exc(), [1., 2.]),
    # list and bc_shape, bc cannot be done
    ([1., 2.], (2, 3), pytest.raises(ValueError), None),
    # element with shape and bc_shape, bc can be done
    (np.ones(1), (2,), no_raised_exc(), [1, 1]),
    # element with shape and bc_shape, bc can be done but shape is the same
    (np.ones(2), (2,), no_raised_exc(), [1., 1.]),
    # element with shape and bc_shape, bc cannot be done
    (np.ones(2), (2, 3), pytest.raises(ValueError), None),

])
def test_sanitize_input(arg, bc_shape, expected_flow, expected_result):
    with expected_flow:
        result = random_variable._sanitize_input(arg, bc_shape)
        print(result)


@pytest.mark.parametrize("inputs, expected_shape", [
    # single basic element
    ([1], ()),
    # multiple basic elements
    ([1, 2, 3], ()),
    # single complex element
    ([np.ones((2, 3))], (2, 3)),
    # multiple complex elements
    ([np.ones((2, 3)), np.ones((2, 6)), np.ones((4, 6))], (4, 6)),
    # multiple simple and complex elements
    ([1, np.ones((2, 6)), 10, np.ones((10, 20))], (10, 20)),
])
def test_maximum_shape(inputs, expected_shape):
    assert np.array_equal(random_variable._maximum_shape(inputs), expected_shape)


def test_random_variable_in_pmodel():
    # test that random variables in pmodel works even if no name has been provided
    @inf.probmodel
    def model():
        inf.Normal(0, 1)

    v = list(model().vars.values())[0]
    assert v.name.startswith('randvar')
    # assert also is_datamodel is false
    assert not v.is_datamodel


def test_random_variable_in_datamodel():
    # test that random variables in datamodel which uses sample_shape warns about that, and uses datamodel size
    with pytest.warns(UserWarning):
        with inf.datamodel(10):
            x = inf.Normal(0, 1, sample_shape=(2,))

        assert x.sample_shape == 10
    # assert also that is_datamodel is true
    assert x.is_datamodel


def test_run_in_session():
    x = inf.Normal(1, 0)
    with tf.Session() as sess:
        assert sess.run(x) == 1
