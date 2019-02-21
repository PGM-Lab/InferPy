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
from collections import OrderedDict
from tensorflow_probability import edward2 as ed
import tensorflow as tf
import warnings

from inferpy import util
from inferpy import exceptions
from . import contextmanager
from .random_variable import RandomVariable


def set_values(**model_kwargs):
    """Creates a value-setting interceptor."""

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]

        return ed.interceptable(f)(*args, **kwargs)

    return interceptor


def probmodel(builder):
    """
    Decorator to create probabilistic models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not needed that the function returns such variables (we capture
    them using ed.tape).
    """
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        tf.reset_default_graph()
        warnings.warn("Provisionally, TF default graph is reset when a prob model is built.")
        return ProbModel(
            builder=lambda: builder(*args, **kwargs)
        )
    return wrapper


class ProbModel:
    """
    Class that implements the probabilistic model functionality.
    It is composed of a graph, capturing the variable relationships, an OrderedDict containing
    the Random Variables in order of creation, and the function which
    """
    def __init__(self, builder):
        # Initialize object attributes
        self.graph = None
        self.vars = None
        self.builder = builder
        # compute vars and graph for this model (NOTE: self.graph and self.vars must exist)
        nx_graph, model_vars = self._build_model()
        # assign computed vars and graph
        self.graph = nx_graph
        self.vars = model_vars

    def _build_model(self):
        # set this graph as active, so datamodel can check and use the model graph
        with contextmanager.prob_model.builder():
            # use edward2 model tape to capture RandomVariable declarations
            with ed.tape() as model_tape:
                self.builder()

            # ed2 RVs created. Relations between them captured in prob_model builder as a networkx graph
            nx_graph = contextmanager.prob_model.get_graph()

            # wrap captured edward2 RVs into inferpy RVs
            model_vars = OrderedDict()
            for k, v in model_tape.items():
                registered_rv = contextmanager.prob_model.get_builder_variable(k)
                if registered_rv is None:
                    # a ed Random Variable. Create a inferpy Random Variable and assign the var directly.
                    # do not know the args and kwars used to build the ed random variable. Use None.
                    model_vars[k] = RandomVariable(v, name=k, is_expanded=False, var_args=None, var_kwargs=None)
                else:
                    model_vars[k] = registered_rv

        return nx_graph, model_vars

    def fit(self, sample_dict):
        # sample_dict must be a non empty python dict
        if not isinstance(sample_dict, dict):
            raise TypeError('The `sample_dict` type must be dict.')
        if len(sample_dict) == 0:
            raise ValueError('The number of mapped variables must be at least 1.')

        # check that all values in dict has the same length (will be the plate size)
        plate_shapes = [util.iterables.get_shape(v) for v in sample_dict.values()]
        plate_sizes = [s[0] if len(s) > 0 else 1 for s in plate_shapes]  # if the shape is (), it is just one element
        plate_size = plate_sizes[0]

        if any(plate_size != x for x in plate_sizes[1:]):
            raise exceptions.InvalidParameterDimension(
                'The number of elements for each mapped variable must be the same.')

        # if the values mapped to randm variables has shape 0, raise an error
        if plate_size == 0:
            raise ValueError('The number of samples in sample_dict must be at least 1.')

        with ed.interception(set_values(**sample_dict)):
            expanded_vars = self._expand_vars(plate_size)

        return expanded_vars

    @util.tf_run_wrapper
    def log_prob(self, sample_dict):
        """ Computes the log probabilities of a (set of) sample(s)"""
        return {k: self.vars[k].log_prob(v) for k, v in sample_dict.items()}

    @util.tf_run_wrapper
    def sum_log_prob(self, sample_dict):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return tf.reduce_sum([tf.reduce_mean(lp) for lp in self.log_prob(sample_dict).values()])

    @util.tf_run_wrapper
    def sample(self, size=1):
        """ Generates a sample for eache variable in the model """
        expanded_vars = self._expand_vars(size)
        return {name: tf.convert_to_tensor(var) for name, var in expanded_vars.items()}

    def _expand_vars(self, size):
        """ Create the expanded model vars using sample_shape as plate size and return the OrderedDict """
        with contextmanager.data_model.fit(size=size):
            _, expanded_vars = self._build_model()
        return expanded_vars
