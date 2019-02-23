import pytest
import numpy as np
import random
import tensorflow as tf


SEED = 607742924


@pytest.fixture()
def tf_reset_default_graph():
    """
    Reset the tf default graph
    """
    tf.reset_default_graph()
    yield


@pytest.fixture()
def reproducible():
    """
    Set tf, numpy and random seed to SEED
    """
    tf.random.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    yield
