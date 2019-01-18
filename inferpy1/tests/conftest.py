import pytest
import tensorflow as tf

SEED = 607742924

@pytest.fixture()
def clear_tf_default_graph():
    tf.reset_default_graph()
    yield
