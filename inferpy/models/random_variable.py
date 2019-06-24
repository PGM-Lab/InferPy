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
import functools
from enum import IntEnum
import tensorflow as tf
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.edward2.interceptor import interceptable

import tensorflow_probability as tfp
from tensorflow_probability.python.edward2 import generated_random_variables
from tensorflow.python.client import session as tf_session
import warnings

from inferpy import contextmanager
from inferpy import util
import inferpy.util.session


# the list of available RandomVariables in edward2. Matches with the available distributions in tensorflow_probability
distributions_all = [rv for rv in generated_random_variables.__all__ if rv != 'as_random_variable']


def _make_edward_random_variable(distribution_obj):
    """Factory function to make random variable given distribution class."""

    @interceptable
    @functools.wraps(distribution_obj, assigned=('__module__', '__name__'))
    def func(*args, **kwargs):
        # pylint: disable=g-doc-args
        """Create a random variable for ${cls}.
        See ${cls} for more details.
        Returns:
          RandomVariable.
        #### Original Docstring for Distribution
        ${doc}
        """
        # pylint: enable=g-doc-args
        sample_shape = kwargs.pop('sample_shape', ())
        value = kwargs.pop('value', None)
        return ed.RandomVariable(distribution=distribution_obj,
                                 sample_shape=sample_shape,
                                 value=value)
    return func


def _try_sess_run(p, sess):
    try:
        return sess.run(p)
    except (RuntimeError, TypeError, ValueError):
        return p


class Kind(IntEnum):
    GLOBAL_HIDDEN = 0
    GLOBAL_OBSERVED = 1
    LOCAL_HIDDEN = 2
    LOCAL_OBSERVED = 3


class RandomVariable:
    """
    Class for random variables. It encapsulates the Random Variable from edward2, and additional properties.

    - It creates a variable generator. It must be a function without parameters, that creates a
      new Random Variable from edward2. It is used to define edward2 models as functions.
      Also, it is useful to define models using the intercept function.

    - The first time the var property is used, it creates a var using the variable generator.
    """

    def __init__(self, var, name, is_datamodel, ed_cls, var_args, var_kwargs, sample_shape,
                 is_observed, observed_value):
        self.var = var
        self.is_datamodel = is_datamodel
        # These parameters are used to allow the re-creation of the random var by build_in_session function
        self._ed_cls = ed_cls
        self._var_args = var_args
        self._var_kwargs = var_kwargs
        self._sample_shape = sample_shape
        self.is_observed = is_observed
        self.observed_value = observed_value

        # if name is provided, use it. Otherwise, use it from var or var.distribution
        if name is None:
            self.name = self.__getattr__('name')
        else:
            self.name = name

    @property
    def type(self):
        first_part = 'LOCAL' if self.is_datamodel else 'GLOBAL'
        second_part = 'OBSERVED' if util.get_session().run(self.is_observed) else 'HIDDEN'

        return getattr(Kind, "{}_{}".format(first_part, second_part))

    def build_in_session(self, sess):
        """
        Allow to build a copy of the random variable but running previously each parameter in the tf session.
        This way, it uses the value of each tf variable or placeholder as a tensor, not as a tf variable or placeholder.
        If this random variable is a ed random variable directly assigned to .var, we cannot re-create it. In this
        case, return self.
        :param sess: tf session used to run each parameter used to build this random variable.
        :returns: the random variable object
        """
        # Cannot re-create the random variable. Return this var itself
        if self._ed_cls is None:
            return self

        # create the ed random variable evaluating each parameter in a tf session
        ed_random_var = self._ed_cls(*[_try_sess_run(a, sess) for a in self._var_args],
                                     **{k: _try_sess_run(v, sess) for k, v in self._var_kwargs.items()},
                                     sample_shape=self._sample_shape)

        initial_value = util.get_session().run(self.observed_value_var)
        is_observed, observed_value = _make_predictable_variables(initial_value, self.name)
        # build the random variable by using the ed random var
        rv = RandomVariable(
            var=ed_random_var,
            name=self.name,
            is_datamodel=self.is_datamodel,
            ed_cls=self._ed_cls,
            var_args=self._var_args,
            var_kwargs=self._var_kwargs,
            sample_shape=self._sample_shape,
            is_observed=self.is_observed,
            observed_value=self.observed_value
        )

        # put the docstring and the name as well as in _make_random_variable function
        docs = RandomVariable.__doc__ + '\n Random Variable information:\n' + \
            ('-' * 30) + '\n' + self._ed_cls.__doc__
        name = self._ed_cls.__name__

        rv.__doc__ += docs
        rv.__name__ = name

        return rv

    def copy(self):
        """
        Makes a of the current random variable where the distribution parameters are fixed.
        :return: new object of class RandomVariable
        """
        return self.build_in_session(inferpy.get_session())

    def __repr__(self):
        # Custom representation of the random variable
        string = "inf.RandomVariable ({} distribution) named {}, shape={}, dtype={}".format(
            self.__name__, self.distribution.name, self.shape, self.dtype.name)

        return "<%s>" % string

    def __getattr__(self, name):
        """
        If try to use attributes or functions not defined for RandomVariables, this function is executed.
        First try to return the same attribute of function from the edward2 RandomVariable.
        Secondly try to return the same element from the distribution object inside the edward2 RandomVariable.
        Otherwise, raise an Exception.
        """
        if hasattr(self.var, name):
            obj = getattr(self.var, name)
            if hasattr(obj, '__call__'):
                return util.tf_run_allowed(obj)
            else:
                return obj
        elif hasattr(self.var.distribution, name):
            obj = getattr(self.var.distribution, name)
            if hasattr(obj, '__call__'):
                return util.tf_run_allowed(obj)
            else:
                return obj
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


def _convert_random_variables_to_tensors(arg):
    # function used to convert Random Variables (from inferpy or edward2) to tensors, even if there
    # are in a list or nested list, in order to allow to use them as arguments for other Random Variables
    if isinstance(arg, (ed.RandomVariable, RandomVariable)):
        return tf.convert_to_tensor(arg)
    elif isinstance(arg, list):
        return [_convert_random_variables_to_tensors(nested_arg) for nested_arg in arg]
    else:
        return arg


def _make_random_variable(distribution_name):
    """Factory function to make random variable given distribution name."""

    distribution_cls = getattr(tfp.distributions, distribution_name)
    ed_random_variable_cls = getattr(ed, distribution_name)

    docs = RandomVariable.__doc__ + '\n Random Variable information:\n' + \
        ('-' * 30) + '\n' + ed_random_variable_cls.__doc__
    name = ed_random_variable_cls.__name__

    def func(*args, **kwargs):
        # The name used to identify the random variable by string
        if 'name' not in kwargs:
            kwargs['name'] = util.name.generate('randvar')
        rv_name = kwargs.get('name')

        if contextmanager.data_model.is_active():
            if 'sample_shape' in kwargs:
                # warn that sampe_shape will be ignored
                warnings.warn('Random Variables defined inside a probabilistic model ignore the sample_shape argument.')
                kwargs.pop('sample_shape', None)
            sample_shape = ()  # used in case that RV is in probmodel, but not in a datamodel
        else:
            # only used if prob model is active
            sample_shape = kwargs.pop('sample_shape', ())

        # convert any Random Variable in a list or nested list in arguments to tensors, allowing to use RV's as arguments
        sanitized_args = [_convert_random_variables_to_tensors(arg) for arg in args]
        sanitized_kwargs = {k: _convert_random_variables_to_tensors(v) for k, v in kwargs.items()}

        # If it is inside a data model, ommit the sample_shape in kwargs if exist and use size from data_model
        # NOTE: Needed here because we need to know the shape of the distribution, as well as its dtype
        # Not using sample shape yet. Used just to create the tensors, and compute the dependencies using the tf graph
        tfp_dist = distribution_cls(*sanitized_args, **sanitized_kwargs)
        if contextmanager.data_model.is_active():
            # create graph once tensors are registered in graph
            contextmanager.randvar_registry.update_graph(rv_name)

            # compute sample_shape now that we have computed the dependencies
            sample_shape = contextmanager.data_model.get_sample_shape(rv_name)

            # create tf.Variable's to allow to observe the Random Variable
            shape = ([sample_shape] if sample_shape else []) + \
                tfp_dist.batch_shape.as_list() + \
                tfp_dist.event_shape.as_list()

            # take into account the dtype of tfp_dist in order to create the initial value correctly
            initial_value = tf.zeros(shape, dtype=tfp_dist.dtype) if shape else tf.constant(0,  dtype=tfp_dist.dtype)

            # build the respective boolean and tf.Variables
            is_observed, observed_value = _make_predictable_variables(initial_value, rv_name)

            # use this context to intercept the value using the tf.cond dependent on the previous tf.Variables
            with ed.interception(util.interceptor.set_values_condition(is_observed, observed_value)):
                ed_random_var = _make_edward_random_variable(tfp_dist)(sample_shape=sample_shape, name=rv_name)

            # if the random variable has been intercepted with other random variable, which also has the tf.Variables,
            # then use their tf.Variables instead of the previously created tf.Variables
            if hasattr(ed_random_var, 'is_observed') and hasattr(ed_random_var, 'observed_value'):
                is_observed = ed_random_var.is_observed
                observed_value = ed_random_var.observed_value

            is_datamodel = True
        else:
            # create tf.Variable's to allow to observe the Random Variable
            shape = tfp_dist.batch_shape.as_list() + tfp_dist.event_shape.as_list()

            # take into account the dtype of tfp_dist in order to create the initial value correctly
            initial_value = tf.zeros(shape, dtype=tfp_dist.dtype) if shape else tf.constant(0,  dtype=tfp_dist.dtype)

            # build the respective boolean and tf.Variables
            is_observed, observed_value = _make_predictable_variables(initial_value, rv_name)

            # use this context to intercept the value using the tf.cond dependent on the previous tf.Variables
            with ed.interception(util.interceptor.set_values_condition(is_observed, observed_value)):
                # sample_shape is sample_shape in kwargs or ()
                ed_random_var = ed_random_variable_cls(*sanitized_args, **sanitized_kwargs, sample_shape=sample_shape)

            # if the random variable has been intercepted with other random variable, which also has the tf.Variables,
            # then use their tf.Variables instead of the previously created tf.Variables
            if hasattr(ed_random_var, 'is_observed') and hasattr(ed_random_var, 'observed_value'):
                is_observed = ed_random_var.is_observed
                observed_value = ed_random_var.observed_value

            is_datamodel = False

        rv = RandomVariable(
            var=ed_random_var,
            name=rv_name,
            is_datamodel=is_datamodel,
            ed_cls=ed_random_variable_cls,
            var_args=sanitized_args,
            var_kwargs=sanitized_kwargs,
            sample_shape=sample_shape,
            is_observed=is_observed,
            observed_value=observed_value,
        )

        # register the variable as it is created. Used to detect dependencies
        contextmanager.randvar_registry.register_variable(rv)
        contextmanager.randvar_registry.update_graph()

        # Doc for help menu
        rv.__doc__ += docs
        rv.__name__ = name

        return rv

    # Doc for help menu
    func.__doc__ = docs
    func.__name__ = name
    return func


def _make_predictable_variables(initial_value, rv_name):
    is_observed = tf.Variable(False, trainable=False,
                              name="inferpy-predict-enabled-{name}".format(name=rv_name or "default"))

    observed_value = tf.Variable(initial_value, trainable=False,
                                 name="inferpy-predict-{name}".format(name=rv_name or "default"))

    util.session.get_session().run(tf.variables_initializer([is_observed, observed_value]))

    return is_observed, observed_value


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
for d in distributions_all:
    globals()[d] = _make_random_variable(d)


def _tensor_conversion_function(rv, dtype=None, name=None, as_ref=False):
    """
        Function that converts the inferpy variable into a Tensor.
        This will enable the use of enable tf.convert_to_tensor(rv)

        If the variable needs to be broadcast_to, do it right now
    """
    # return the tf.Variable last snapshot if it is observed, and the ed2 evaluation (ed2.value) otherwise
    return tf.convert_to_tensor(rv.observed_value.value() if util.get_session().run(rv.is_observed) else rv.var)


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
