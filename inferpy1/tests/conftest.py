import pytest
import tensorflow as tf

SEED = 607742924

from inferpy.util import get_default_session

@pytest.fixture()
def close_tf_default_session():
    """
    From the tensorflow function `reset_default_graph` doc: `https://www.tensorflow.org/api_docs/python/tf/reset_default_graph`:
    ```
    NOTE: The default graph is a property of the current thread. This function applies only to the current thread.
    Calling this function while a tf.Session or tf.InteractiveSession is active will result in undefined behavior.
    Using any previously created tf.Operation or tf.Tensor objects after calling this function will result in undefined behavior.
    ```
    Therefore, instead of calling tf.reset_default_graph, we close the default session. If required, a new Interactive Session will be created on demand by the util.runtime module.
    """
    get_default_session().close()
    yield
    