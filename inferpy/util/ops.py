import numpy as np
import inferpy.models
import tensorflow as tf
import collections
import six



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



def case_states(var, d, default=None, exclusive=True, strict=False, name='case'):
    out_d = {}


    def f(p): return tf.constant(p)


    for s, p in six.iteritems(d):

        out_d.update({tf.reduce_all(tf.equal(var.dist, tf.constant(s))): (lambda pp : lambda: f(pp))(p)})

    return tf.case(out_d, default=default, exclusive=exclusive,strict=strict,name=name)



def case(d, default=None, exclusive=True, strict=False, name='case'):
    out_d = {}


    def f(p): return tf.constant(p)

    for c, p in six.iteritems(d):

        out_d.update({tf.reduce_all(tf.equal(c.base_object, True)): (lambda pp : lambda: f(pp))(p)})


    if default != None:
        default = (lambda pp : lambda: f(pp))(default)

    return tf.case(out_d, default=default, exclusive=exclusive,strict=strict,name=name)



def matmul(A,B):
    return A.__matmul__(B)