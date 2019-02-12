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
import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.edward2 import generated_random_variables
from tensorflow.python.client import session as tf_session
import warnings

from inferpy.util import tf_run_eval
from . import contextmanager
from inferpy import exceptions


rv_all = generated_random_variables.rv_all  # the list of available RandomVariables in edward2


class RandomVariable:
    """
    Class for random variables. It encapsulares the Random Variable from edward2, and additional properties.

    - It creates a variable generator. It must be a function without parameters, that creates a
      new Random Variable from edward2. It is used to define edward2 models as functions.
      Also, it is useful to define models using the intercept function.

    - The first time the var property is used, it creates a var using the variable generator.
    """

    def __init__(self, var, name, is_expanded, is_datamodel):
        self.var = var
        self.is_expanded = is_expanded
        self.is_datamodel = is_datamodel

        # if name is provided, use it. Otherwise, use it from var or var.distribution
        if name is None:
            self.name = self.__getattr__('name')
        else:
            self.name = name

    # If try to use attributes or functions not defined for RandomVariables, this function is executed.
    # First try to return the same attribute of function from the edward2 RandomVariable.
    # Secondly try to return the same element from the distribution object inside the edward2 RandomVariable.
    # Otherwise, raise an Exception.
    def __getattr__(self, name):
        if hasattr(self.var, name):
            return tf_run_eval(getattr(self.var, name))
        elif hasattr(self.var.distribution, name):
            return tf_run_eval(getattr(self.var.distribution, name))
        else:
            raise AttributeError('Property or method "{name}" not implemented in "{classname}"'.format(
                name=name,
                classname=type(self.var).__name__
            ))

    # In the following, define the operations by using the edward2 operator definition

    def __add__(self, other):
        return _operator(self.var, other, '__add__')

    def __radd__(self, other):
        return _operator(self.var, other, '__radd__')

    def __sub__(self, other):
        return _operator(self.var, other, '__sub__')

    def __rsub__(self, other):
        return _operator(self.var, other, '__rsub__')

    def __mul__(self, other):
        return _operator(self.var, other, '__mul__')

    def __rmul__(self, other):
        return _operator(self.var, other, '__rmul__')

    def __div__(self, other):
        return _operator(self.var, other, '__div__')

    def __rdiv__(self, other):
        return _operator(self.var, other, '__rdiv__')

    def __truediv__(self, other):
        return _operator(self.var, other, '__truediv__')

    def __rtruediv__(self, other):
        return _operator(self.var, other, '__rtruediv__')

    def __floordiv__(self, other):
        return _operator(self.var, other, '__floordiv__')

    def __rfloordiv__(self, other):
        return _operator(self.var, other, '__rfloordiv__')

    def __mod__(self, other):
        return _operator(self.var, other, '__mod__')

    def __rmod__(self, other):
        return _operator(self.var, other, '__rmod__')

    def __lt__(self, other):
        return _operator(self.var, other, '__lt__')

    def __le__(self, other):
        return _operator(self.var, other, '__le__')

    def __gt__(self, other):
        return _operator(self.var, other, '__gt__')

    def __ge__(self, other):
        return _operator(self.var, other, '__ge__')

    def __and__(self, other):
        return _operator(self.var, other, '__and__')

    def __rand__(self, other):
        return _operator(self.var, other, '__rand__')

    def __or__(self, other):
        return _operator(self.var, other, '__or__')

    def __ror__(self, other):
        return _operator(self.var, other, '__ror__')

    def __xor__(self, other):
        return _operator(self.var, other, '__xor__')

    def __rxor__(self, other):
        return _operator(self.var, other, '__rxor__')

    def __getitem__(self, *args):
        return self.var.__getitem__(*args)

    def __pow__(self, other):
        return _operator(self.var, other, '__pow__')

    def __rpow__(self, other):
        return _operator(self.var, other, '__rpow__')

    def __invert__(self):
        return self.var.__invert__()

    def __neg__(self):
        return self.var.__neg__()

    def __abs__(self):
        return self.var.__abs__()

    def __matmul__(self, other):
        return _operator(self.var, other, '__matmul__')

    def __rmatmul__(self, other):
        return _operator(self.var, other, '__rmatmul__')

    def __iter__(self):
        return self.var.__iter__()

    def __bool__(self):
        return self.var.__bool__()

    def __nonzero__(self):
        return self.var.__nonzero__()


def _sanitize_input(arg, bc_shape):
    if bc_shape is not None and (isinstance(arg, list) or hasattr(arg, 'shape')):
        # This items are used for sure as RV parameters (only sample_shape can interfer, and has been removed)
        # For each arg, try to tf.broadcast_to bc_shape, and convert to a single tensor using tf.stack
        if _get_input_shape(arg) != bc_shape:
            # Try to broadcast to bc_shape. If exception, use arg to stack (i.e. all are simple scalars)
            try:
                # broadcast each element to the bc_shape
                bc_arg = tf.broadcast_to(arg, bc_shape)
            except ValueError:
                # if broadcast fails, raise custom error
                raise exceptions.InvalidParameterDimension('Parameters cannot be broadcasted. Check their shapes.')
        else:
            bc_arg = arg

        return tf.stack(bc_arg, axis=0)
    else:
        # if it is a dict, other objects return arg as it is (can be other input accepted by ed.RandomVariable).
        return arg


def _get_input_shape(x):
    # get the shape of an element x. If it is an element with a shape attribute, return it. If it is a list,
    # compute the shape by checking the len, and the shape of internal elements. In that case, the shape must
    # be consistent. Finally, in other case return () as shape.
    if isinstance(x, list):
        shapes = [_get_input_shape(subx) for subx in x]
        if any([s != shapes[0] for s in shapes[1:]]):
            raise exceptions.InvalidParameterDimension('Parameter dimension not consistent: {}'.format(x))
        return (len(x), ) + shapes[0]
    else:
        if hasattr(x, 'shape'):
            return tuple(x.shape)
        else:
            return ()


def _maximum_shape(list_inputs):
    shapes = [_get_input_shape(x) for x in list_inputs]
    # get the shape with maximum number of elements
    [s for s in shapes]
    idx = np.argmax([np.multiply.reduce(s) if len(s) > 0 else 0 for s in shapes])
    return shapes[idx]


def _make_random_variable(distribution_cls):
    """Factory function to make random variable given distribution class."""
    docs = RandomVariable.__doc__ + '\n Random Variable information:\n' + ('-' * 30) + '\n' + distribution_cls.__doc__
    name = distribution_cls.__name__

    def func(*args, **kwargs):
        rv_name = kwargs.get('name', None)

        # compute maximum shape between shapes of inputs, and apply broadcast to the smallers in _sanitize_input
        max_shape = _maximum_shape(args + tuple(kwargs.values()))

        if not contextmanager.prob_model.is_active():
            contextmanager.prob_model.activate_default()

        try:

            if contextmanager.prob_model.is_active():
                # At this point, the name argument MUST be declared if prob_model is active
                if 'name' not in kwargs:
                    raise exceptions.NotNamedRandomVariable(
                        'Random Variables defined inside a probabilistic model must have a name.')
                if 'sample_shape' in kwargs:
                    # warn that sampe_shape will be ignored
                    warnings.warn('Random Variables defined inside a probabilistic model ignore the sample_shape argument.')
                    kwargs.pop('sample_shape', None)
            else:
                # only used if not prob_model.is_active()
                sample_shape = kwargs.pop('sample_shape', ())

            # sanitize will consist on tf.stack list, and each element must be broadcast_to to match the shape
            sanitized_args = [_sanitize_input(arg, max_shape) for arg in args]
            sanitized_kwargs = {k: _sanitize_input(v, max_shape) for k, v in kwargs.items()}

            # If it is inside a prob model, ommit the sample_shape in kwargs if exist and use size from data_model
            if contextmanager.prob_model.is_active():
                # Not using sample shape yet. Used just to create the tensors, and
                # compute the dependencies by using the tf graph
                ed_random_var = distribution_cls(*sanitized_args, **sanitized_kwargs)

                # create graph once tensors are registered in graph
                contextmanager.prob_model.update_graph(rv_name)

                # compute sample_shape now that we have computed the dependencies
                sample_shape, is_expanded = contextmanager.data_model.get_random_variable_shape(args, kwargs)
                ed_random_var._sample_shape = sample_shape
                is_datamodel = True
            else:
                # sample_shape is sample_shape in kwargs or ()
                is_expanded = False
                is_datamodel = False
                ed_random_var = distribution_cls(*sanitized_args, **sanitized_kwargs, sample_shape=sample_shape)

            rv = RandomVariable(
                var=ed_random_var,
                name=rv_name,
                is_expanded=is_expanded,
                is_datamodel=is_datamodel
            )

            if is_datamodel:
                # inside prob models, register the variable as it is created. Used for prob model builder context
                contextmanager.prob_model.register_variable(rv)

            # Doc for help menu
            rv.__doc__ += docs
            rv.__name__ = name

        finally:
            ##### if using the default prob_model, this should be deactivate
            if contextmanager.prob_model.is_default():
                contextmanager.prob_model.deactivate_default()

        return rv

    # Doc for help menu
    func.__doc__ = docs
    func.__name__ = name
    return func


def _operator(var, other, operator_name):
    """
    Function to apply the operation called `operator_name` by using a RandomVariable
    object and other object (which can be a RandomVariable or not).
    """
    # get the operator_name function from the edward2 variable `var`
    # and call that function using the edward2 object if other is a RandomVariable,
    # otherwise just use `other` as argument
    return getattr(var, operator_name)(other.var if isinstance(other, RandomVariable) else other)


# Define all Random Variables existing in edward2
for rv in rv_all:
    globals()[rv] = _make_random_variable(getattr(ed, rv))


def _tensor_conversion_function(rv, dtype=None, name=None, as_ref=False):
    """
        Function that converts the inferpy variable into a Tensor.
        This will enable the use of enable tf.convert_to_tensor(rv)

        If the variable needs to be broadcast_to, do it right now
    """
    return tf.convert_to_tensor(rv.var)


# register the conversion function into a tensor
tf.register_tensor_conversion_function(  # enable tf.convert_to_tensor
    RandomVariable, _tensor_conversion_function)


def _session_run_conversion_fetch_function(rv):
    """
        This will enable run and operations with other tensors
    """
    return ([tf.convert_to_tensor(rv)], lambda val: val[0])


tf_session.register_session_run_conversion_functions(  # enable sess.run, eval
    RandomVariable,
    _session_run_conversion_fetch_function)
