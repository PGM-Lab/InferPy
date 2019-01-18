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
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.edward2 import generated_random_variables

from inferpy.util import tf_run_eval


rv_all = generated_random_variables.rv_all


class RandomVariable:
    """
    Class for random variables. It encapsulares the Random Variable from edward2, and additional properties.

    - It creates a variable generator. It must be a function without parameters, that creates a
      new Random Variable from edward2. It is used to define edward2 models as functions.
      Also, it is useful to define models using the intercept function.

    - The first time the var property is used, it creates a var using the variable generator.
    """

    def __init__(self, var_generator, *args, is_observed=False, **kwargs):
        self.is_observed = is_observed
        self._var_generator = var_generator
        self._var = None  # initially None. Created automatically the first time it is called.

    @property
    def var(self):
        # it is not 
        if self._var is None:
            self._var = self._var_generator()
        return self._var

    @property
    def name(self):
        """ name of the variable"""
        ed_name = self.var.distribution.name
        return ed_name[:-1]

    def __getattr__(self, name):
        if hasattr(self.var, name):
            return tf_run_eval(getattr(self.var, name))
        elif hasattr(self.var.distribution, name):
            return tf_run_eval(getattr(self.var.distribution, name))
        else:
            raise NotImplementedError('Property or method "{name}" not implemented in "{classname}"'.format(
                name=name,
                classname=type(self.var).__name__
            ))

    def __add__(self, other):
        return _operator(self.var.__add__, other)
    
    def __radd__(self, other):
        return _operator(self.var.__radd__, other)


def _make_random_variable(distribution_cls):
    """Factory function to make random variable given distribution class."""
    docs = RandomVariable.__doc__ + '\n Random Variable information:\n' + ('-' * 30) + '\n' + distribution_cls.__doc__
    name = distribution_cls.__name__

    def func(*args, **kwargs):
        rv = RandomVariable(var_generator=lambda: distribution_cls(*args, **kwargs))
        # For help menu
        rv.__doc__ += docs
        rv.__name__ = name
        return rv
    # For help menu
    func.__doc__ = docs
    func.__name__ = name
    return func


def _operator(operator, *args):
    return RandomVariable(lambda: ed.Deterministic(operator(*args)))


# Define all Random Variables existing in edward2
for rv in rv_all:
    globals()[rv] = _make_random_variable(getattr(ed, rv))
