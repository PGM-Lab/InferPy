import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import pytest

from inferpy import models

@pytest.mark.parametrize("model_object", [
    # Simple random variable using scalars as parameters
    (models.Normal(0, 1)),
    # Simple random variable using a list as parameter
    (models.Normal([0, 0, 0, 0], 1)),
    # Simple random variable using a numpy array as parameter
    (models.Normal(np.zeros(5), 1)),
    # Simple random variable using a tensor as parameter
    (models.Normal(0, tf.ones(5))),
    # Simple random variable using another random variable as parameter
    #(models.Normal(models.Normal(0, 1), 1)),
    # Simple random variable using a combination of the previously tested options as parameter
    #(models.Normal([models.Normal(0, 1), 1, tf.constant(1.)], 1)),
    # Random variable operation
    #(models.Normal(0, 1) + models.Normal(0, 1)),
])
def test_edward_type(clear_tf_default_graph, model_object):
    assert isinstance(model_object.var, ed.RandomVariable)


def test_is_observed(clear_tf_default_graph):
    x = models.Normal(0, 1)
    assert not x.is_observed
    x = models.Normal(0, 1, is_observed=True)
    assert x.is_observed

    # test setter
    x.is_observed = False
    assert not x.is_observed


def test_name(clear_tf_default_graph):
    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo'
    
    x = models.Normal(0, 1, name='foo')
    assert x.name == 'foo_1'
    
    # Automatic name generation. Internal final / is not shown
    x = models.Normal(0, 1)
    assert isinstance(x.name, str)
    assert x.name[-1] != '/'
