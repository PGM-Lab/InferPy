import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import pytest

from inferpy import models
from inferpy.models.random_variable import _sanitize_args
from inferpy.models.random_variable import _sanitize_kwargs

from inferpy.util import tf_sess

"""
clear_tf_default_graph is not used for the functions where Random Variables
are created in the parametrize decorator, because these Tensors are created
in the default graph, and if it is reset then operations might fail.
"""


@pytest.mark.parametrize("list_args, expected_classes", [
    # Empty list
    ([], []),
    # Simple inferpy rv
    ([models.Normal(0, 1)], [ed.RandomVariable]),
    # Simple edward2 rv
    ([ed.Normal(0, 1)], [ed.RandomVariable]),
    # Simple tensor
    ([tf.constant(0.)], [tf.Tensor]),
    # Simple number
    ([0.], [float]),
    # Combination of all the previous elements
    ([models.Normal(0, 1), ed.Normal(0, 1), tf.constant(0.), 0.],
     [ed.RandomVariable, ed.RandomVariable, tf.Tensor, float]
     ),
    # Nested combinaton of elements
    ([[models.Normal(0, 1), ed.Normal(0, 1)], ed.Normal(0, 1)],
     [tf.Tensor, ed.RandomVariable]
     ),

])
def test_sanitize_args(list_args, expected_classes):
    clean_args = _sanitize_args(list_args)
    for ca, expected_ca in zip(clean_args, expected_classes):
        assert isinstance(ca, expected_ca)


@pytest.mark.parametrize("dict_args, expected_classes", [
    # Empty dict
    (dict(), dict()),
    # Simple inferpy rv
    (dict(loc=models.Normal(0, 1)), dict(loc=ed.RandomVariable)),
    # Simple edward2 rv
    (dict(loc=ed.Normal(0, 1)), dict(loc=ed.RandomVariable)),
    # Simple tensor
    (dict(loc=tf.constant(0.)), dict(loc=tf.Tensor)),
    # Simple number
    (dict(loc=0.), dict(loc=float)),
    # Combination of all the previous elements
    (dict(loc=models.Normal(0, 1), scale=ed.Normal(0, 1), alpha=tf.constant(0.), beta=0.),
     dict(loc=ed.RandomVariable, scale=ed.RandomVariable, alpha=tf.Tensor, beta=float)
     ),
    # Nested combinaton of elements
    (dict(loc=[models.Normal(0, 1), ed.Normal(0, 1)], scale=ed.Normal(0, 1)),
     dict(loc=tf.Tensor, scale=ed.RandomVariable)
     ),

])
def test_sanitize_kwargs(dict_args, expected_classes):
    clean_args = _sanitize_kwargs(dict_args)
    for k, v in clean_args.items():
        assert isinstance(v, expected_classes[k])


def test_operations(clear_tf_default_graph):
    # TODO: test all operations using parametrize
    # Use both inferpy Random Variables
    x = models.Normal([0., 1.], 1)
    y = models.Normal([1., 2.], 2)
    assert isinstance(abs(x), tf.Tensor)
    assert isinstance(x[1], tf.Tensor)
    assert isinstance((x + y), tf.Tensor)
    assert isinstance((x * y), tf.Tensor)
    assert isinstance((x / y), tf.Tensor)
    assert isinstance((x ** y), tf.Tensor)

    # Use inferpy and edward2 Random Variables
    x = models.Normal([0., 1.], 1)
    y = ed.Normal([1., 2.], 2)
    assert isinstance((x + y), tf.Tensor)
    assert isinstance((x * y), tf.Tensor)
    assert isinstance((x / y), tf.Tensor)
    assert isinstance((x ** y), tf.Tensor)

    # Use inferpy Random Variables and tensors
    x = models.Normal([0., 1.], 1)
    y = tf.constant(1.)
    assert isinstance((x + y), tf.Tensor)
    assert isinstance((x * y), tf.Tensor)
    assert isinstance((x / y), tf.Tensor)
    assert isinstance((x ** y), tf.Tensor)


@pytest.mark.parametrize("model_object", [
    # Simple random variable using scalars as parameters
    (models.Normal(0, 1)),
    # Simple random variable using a list as parameter
    (models.Normal([0., 0., 0., 0.], 1)),
    # Simple random variable using a numpy array as parameter
    (models.Normal(np.zeros(5), 1)),
    # Simple random variable using a tensor as parameter
    (models.Normal(0, tf.ones(5))),
    # Simple random variable using another random variable as parameter
    (models.Normal(models.Normal(0, 1), 1)),
    # Simple random variable using a combination of the previously tested options as parameter
    (models.Normal([models.Normal(0, 1), 1., tf.constant(1.)], 1.)),
    # Random variable operation used to define a Random Variable
    (models.Normal(models.Normal(0, 1) + models.Normal(0, 1), 1)),
])
def test_edward_type(clear_tf_default_graph, model_object):
    assert isinstance(model_object.var, ed.RandomVariable)


def test_name(clear_tf_default_graph):
    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo'

    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo_1'

    # Automatic name generation. Internal final / is not shown
    x = models.Normal(0, 1)
    assert isinstance(x.name, str)
    assert x.name[-1] != '/'


def test_tensor_register(clear_tf_default_graph):
    x = models.Normal(0, 1, name='foo')

    # This is basically x.sample(), so there is no need to implement it
    # x.eval()

    assert tf_sess.run(x) == 5
    assert isinstance(tf.convert_to_tensor(x), tf.Tensor)
    assert tf.convert_to_tensor(x).eval() == 5
    # assert (tf.constant(5.) + x).eval() == 10  # "_as_graph_element" not implemented in "RandomVariable"
    assert (x + tf.constant(5.)).eval() == 10
