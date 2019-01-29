import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import pytest

from inferpy import models

from inferpy.util import get_default_session


def test_operations(close_tf_default_session):
    # TODO: test all operations using parametrize
    # Use both inferpy Random Variables
    x = models.Normal([0., 1.], 1)
    y = models.Normal([1., 2.], 2)
    assert isinstance(abs(x), tf.Tensor)
    assert isinstance(x[1], tf.Tensor)
    assert isinstance((x + y), tf.Tensor)
    assert isinstance((y * x), tf.Tensor)
    assert isinstance((x / y), tf.Tensor)
    assert isinstance((y ** x), tf.Tensor)

    # Use inferpy and edward2 Random Variables
    x = models.Normal([0., 1.], 1)
    y = ed.Normal([1., 2.], 2)
    assert isinstance((y + x), tf.Tensor)
    assert isinstance((x * y), tf.Tensor)
    assert isinstance((y / x), tf.Tensor)
    assert isinstance((x ** y), tf.Tensor)

    # Use inferpy Random Variables and tensors
    x = models.Normal([0., 1.], 1)
    y = tf.constant(1.)
    assert isinstance((x + y), tf.Tensor)
    assert isinstance((y * x), tf.Tensor)
    assert isinstance((x / y), tf.Tensor)
    assert isinstance((y ** x), tf.Tensor)


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
def test_edward_type(close_tf_default_session, model_object):
    assert isinstance(model_object.var, ed.RandomVariable)


def test_name(close_tf_default_session):
    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo'

    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo_1'

    # Automatic name generation. Internal final / is not shown
    x = models.Normal(0, 1)
    assert isinstance(x.name, str)
    assert x.name[-1] != '/'


def test_tensor_register(close_tf_default_session):
    # This allows to run a inferpy.models.RandomVariable directly in a tf session.
    
    x = models.Normal(5., 0., name='foovar')

    assert get_default_session().run(x)
    assert isinstance(tf.convert_to_tensor(x), tf.Tensor)
    assert tf.convert_to_tensor(x).eval() == 5.
    assert (tf.constant(5.) + x).eval() == 10.
    assert (x + tf.constant(5.)).eval() == 10.
