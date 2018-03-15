import numpy as np
import inferpy.models
import tensorflow as tf
import collections



def get_total_dimension(x):

    D = 0

    if np.ndim(x) == 0:
        x = [x]

    for xi in x:
        if np.isscalar(xi):
            D = D + 1
        elif isinstance(xi, inferpy.models.RandomVariable):
            D = D + xi.dim
        elif isinstance(xi, tf.Tensor):
            D = D + xi.get_shape().as_list()[-1]

        else:
            raise ValueError("Wrong input type")


    return D




def param_to_tf(x):
    if np.isscalar(x):
        return tf.constant(x, dtype="float32")
    elif isinstance(x, inferpy.models.RandomVariable):
        return x.base_object
    else:
        raise ValueError("wrong input value in param_to_tf")


def ndim(v):
    if np.isscalar(v):
        return 0
    if not isinstance(v, collections.Iterable):
        v = [v]

    out = [1 if np.isscalar(x)
                  else (ndim(x) + 1 if type(x) in [np.ndarray, list]
                        else ndim(x.sample(1)[0]))
                  for x in v]
    return np.max(out)
