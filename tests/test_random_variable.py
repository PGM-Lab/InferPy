import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import pytest

import inferpy as inf
from inferpy.models import random_variable


# TODO: boolean operators and iter are not tested
@pytest.mark.parametrize("tensor, expected", [
    # sum
    ("inf.Normal([0., 1.], 0) + inf.Normal([1., 2.], 0)", [1., 3.]),
    ("[0, 1] + inf.Normal([1., 2.], 0)", [1., 3.]),
    ("inf.Normal([0., 1.], 0) + [1, 2]", [1., 3.]),
    # sub
    ("inf.Normal([0., 1.], 0) - inf.Normal([1., 2.], 0)", [-1., -1.]),
    ("[0, 1] - inf.Normal([1., 2.], 0)", [-1., -1.]),
    ("inf.Normal([0., 1.], 0) - [1, 2]", [-1., -1.]),
    # mult
    ("inf.Normal([0., 1.], 0) * inf.Normal([1., 2.], 0)", [0., 2.]),
    ("[0, 1] * inf.Normal([1., 2.], 0)", [0., 2.]),
    ("inf.Normal([0., 1.], 0) * [1, 2]", [0., 2.]),
    # div
    ("inf.Normal([0., 1.], 0) / inf.Normal([1., 2.], 0)", [0., 0.5]),
    ("[0, 1] / inf.Normal([1., 2.], 0)", [0., 0.5]),
    ("inf.Normal([0., 1.], 0) / [1, 2]", [0., 0.5]),
    # mod
    ("inf.Normal([0., 1.], 0) % inf.Normal([2., 2.], 0)", [0., 1.]),
    ("[0, 1] % inf.Normal([2., 2.], 0)", [0., 1.]),
    ("inf.Normal([0., 1.], 0) % [2, 2]", [0., 1.]),
    # lt
    ("inf.Normal(0., 0) < inf.Normal(2., 0)", True),
    ("1 < inf.Normal(0., 0)", False),
    ("inf.Normal(1., 0) < 2", True),
    # le
    ("inf.Normal(2., 0) <= inf.Normal(2., 0)", True),
    ("1 <= inf.Normal(0., 0)", False),
    ("inf.Normal(1., 0) <= 1", True),
    # gt
    ("inf.Normal(5., 0) > inf.Normal(2., 0)", True),
    ("1 > inf.Normal(3., 0)", False),
    ("inf.Normal(3., 0) > 1", True),
    # getitem
    ("inf.Normal([0., 1., 2., 3.], 0)[0]", 0),
    ("inf.Normal([0., 1., 2., 3.], 0)[2]", 2),
    ("inf.Normal(tf.ones((3, 2)), 0)[2][0]", 1),
    # pow
    ("inf.Normal(2., 0) ** inf.Normal(3., 0)", 8.),
    ("3 ** inf.Normal(2., 0)", 9.),
    ("inf.Normal(3., 0) ** 3", 27.),
    # neg
    ("-inf.Normal(5., 0)", -5),
    ("-inf.Normal(-5., 0)", 5),
    # abs
    ("abs(inf.Normal(5., 0))", 5),
    ("abs(inf.Normal(-5., 0))", 5),
    # matmul
    ("tf.matmul(inf.Normal(tf.ones((2, 3), dtype=np.float32), 0), inf.Normal(tf.eye(3, dtype=np.float32), 0))",
     np.ones((2, 3))),
    ("tf.matmul(np.ones((2, 3), dtype=np.float32), inf.Normal(tf.eye(3, dtype=np.float32), 0))",
     np.ones((2, 3))),
    ("tf.matmul(inf.Normal(np.ones((2, 3), dtype=np.float32), 0), np.eye(3, dtype=np.float32))",
     np.ones((2, 3))),
])
def test_operations(tensor, expected):
    result = inf.get_session().run(eval(tensor))
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


def test_convert_random_variables_to_tensors():
    # element without RVs
    element = 1
    element = random_variable._convert_random_variables_to_tensors(element)
    assert element == element

    # list of elements different from RVs
    element = [1, 1]
    element = random_variable._convert_random_variables_to_tensors(element)
    assert element == element

    # numpy array of elements different from RVs
    element = np.ones((3, 2))
    element = random_variable._convert_random_variables_to_tensors(element)
    assert (element == element).all()

    # A single Random Variable
    element = inf.Normal(0, 1)
    result = random_variable._convert_random_variables_to_tensors(element)
    assert isinstance(element, random_variable.RandomVariable) and not isinstance(result, random_variable.RandomVariable)

    # A list with some Random Variables
    element = [inf.Normal(0, 1), 1, inf.Normal(0, 1), 2]
    result = random_variable._convert_random_variables_to_tensors(element)
    assert all([
        isinstance(element[i], random_variable.RandomVariable) and
        not isinstance(result[i], random_variable.RandomVariable)
        for i in [0, 2]])

    # A list with some nested Random Variables
    element = [[inf.Normal(0, 1), 1, inf.Normal(0, 1), 2]]
    result = random_variable._convert_random_variables_to_tensors(element)
    assert all([
        isinstance(element[0][i], random_variable.RandomVariable) and
        not isinstance(result[0][i], random_variable.RandomVariable)
        for i in [0, 2]])


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
    inf.get_session()
    x = inf.Normal(1, 0)
    assert inf.get_session().run(x) == 1
