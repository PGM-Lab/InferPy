# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


""" Module implementing some useful operations over tensors and random variables """




import numpy as np
import inferpy.models
import tensorflow as tf
import collections
import six




def param_to_tf(x):

    """ Transforms either a scalar or a random variable into a Tensor"""


    if np.isscalar(x):
        return tf.constant(x, dtype="float32")
    elif isinstance(x, inferpy.models.RandomVariable):
        return x.base_object
    else:
        raise ValueError("wrong input value in param_to_tf")



def case_states(var, d, default=None, exclusive=True, strict=False, name='case'):

    """ Control flow operation depending of the outcome of a discrete variable.

    Internally, the operation tensorflow.case is invoked. Unlike the tensorflow operation, this one
    accepts InferPy variables as input parameters.

    Args:
        var: Control InferPy discrete random variable.
        d : dictionary where the keys are each of the possible values of control variable
        and the values are returning tensors for each case.
        exclusive: True iff at most one case is allowed to evaluate to True.
        name: name of the resulting tensor.

    Return:
        Tensor implementing the case operation. This is the output of the operation
        tensorflow.case internally invoked.


    """

    out_d = {}

    if not isinstance(var, inferpy.models.RandomVariable):
        var = inferpy.models.Deterministic(var)


    def f(p): return tf.constant(p)


    for s, p in six.iteritems(d):

        out_d.update({tf.reduce_all(tf.equal(var.dist, tf.constant(s))): (lambda pp : lambda: f(pp))(p)})

    return tf.case(out_d, default=default, exclusive=exclusive,strict=strict,name=name)



def case(d, default=None, exclusive=True, strict=False, name='case'):

    """ Control flow operation depending of the outcome of a tensor. Any expression
    in tensorflow giving as a result a boolean is allowed as condition.

    Internally, the operation tensorflow.case is invoked. Unlike the tensorflow operation, this one
    accepts InferPy variables as input parameters.

    Args:
        d : dictionary where the keys are the conditions (i.e. boolean tensor).
        exclusive: True iff at most one case is allowed to evaluate to True.
        name: name of the resulting tensor.

    Return:
        Tensor implementing the case operation. This is the output of the operation
        tensorflow.case internally invoked.


    """


    out_d = {}

    def f(p): return tf.constant(p)

    for c, p in six.iteritems(d):

        out_d.update({tf.reduce_all(tf.equal(c.base_object, True)): (lambda pp : lambda: f(pp))(p)})


    if default != None:
        default = (lambda pp : lambda: f(pp))(default)

    return tf.case(out_d, default=default, exclusive=exclusive,strict=strict,name=name)



def gather(
        params,
        indices,
        validate_indices=None,
        name=None,
        axis=0 ):
    """ Operation for selecting some of the items in a tensor.

    Internally, the operation tensorflow.gather is invoked. Unlike the tensorflow operation, this one
    accepts InferPy variables as input parameters.

    Args:
        params: A Tensor. The tensor from which to gather values. Must be at least rank axis + 1.
        indices: A Tensor. Must be one of the following types: int32, int64. Index tensor. Must be in range
        [0, params.shape[axis]).
        axis: A Tensor. Must be one of the following types: int32, int64. The axis in params to gather indices
        from. Defaults to the first dimension. Supports negative indexes.
        name: A name for the operation (optional).



    Return:
        A Tensor. Has the same type as params.. This is the output of the operation
        tensorflow.gather internally invoked.


    """


    tf_params = params.base_object if isinstance(params, inferpy.models.RandomVariable)==True else params
    tf_indices = indices.base_object if isinstance(indices, inferpy.models.RandomVariable) == True else indices

    return  tf.gather(tf_params, tf_indices, validate_indices, name, axis)


def matmul(
        a,
        b,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
        name=None):


    """ Matrix multiplication.

    Input objects may be tensors but also InferPy variables.

    Args:
        a: Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.
        b: Tensor with same type and rank as a.
        transpose_a: If True, a is transposed before multiplication.
        transpose_b: If True, b is transposed before multiplication.
        adjoint_a: If True, a is conjugated and transposed before multiplication.
        adjoint_b: If True, b is conjugated and transposed before multiplication.
        a_is_sparse: If True, a is treated as a sparse matrix.
        b_is_sparse: If True, b is treated as a sparse matrix.
        name: Name for the operation (optional).

    Retruns:
        An InferPy variable of type Deterministic encapsulating the resulting tensor
        of the multiplications.


    """

    res = inferpy.models.Deterministic()


    a_shape = shape_to_list(a)
    b_shape = shape_to_list(b)


    if isinstance(a, inferpy.models.RandomVariable):
        a = a.base_object

    if isinstance(b, inferpy.models.RandomVariable):
        b = b.base_object


    a = a if len(a_shape) > 1 else tf.reshape(a, [1] + a_shape)
    b = b if len(b_shape) > 1 else tf.reshape(b, [1] + b_shape)

    res.base_object = tf.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, name)

    return res


def dot(x,y):

    """ Compute dot product between an InferPy or Tensor object. The number of batches N equal to 1
    for one of them, and higher for the other one.

     If necessarily, the order of the operands may be changed.

     Args:
         x: first operand. This could be an InferPy variable, a Tensor, a numpy object or a numeric Python list.
         x: second operand. This could be an InferPy variable, a Tensor, a numpy object or a numeric Python list.


    Retruns:
        An InferPy variable of type Deterministic encapsulating the resulting tensor
        of the multiplications.

     """


    x_shape = shape_to_list(x)
    y_shape = shape_to_list(y)

    if len(x_shape) == 1 and len(y_shape)==2:

        a = y
        b = x

    elif len(x_shape) == 2 and len(y_shape) == 1:
        a = x
        b = y


    else:
        raise ValueError("Wrong dimensions")


    return matmul(a, b, transpose_b=True)



def shape_to_list(a):

    """ Transforms the shape of an object into a list

    Args:
        a : object whose shape will be transformed. This could be an InferPy variable, a Tensor, a numpy object or a numeric Python list.

    """

    if isinstance(a, inferpy.models.RandomVariable):
        a_shape = a.shape
    elif isinstance(a, np.ndarray):
        a_shape = list(a.shape)
    elif isinstance(a, list):
        a_shape = list(np.shape(a))
    elif isinstance(a, tf.Tensor):
        a_shape = a._shape_as_list()
    else:
        raise ValueError("Wrong input type "+a)

    return a_shape


def fix_shape(s):

    """ Transforms a shape list into a standard InferPy shape format. """

    ret = []

    for i in range(0,len(s)):
        if i in [0, len(s)-1] or s[i] != 1:
            ret.append(s[i])

    if len(ret) == 0:
        return [1]

    return ret






