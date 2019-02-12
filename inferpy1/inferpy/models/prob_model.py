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

from inferpy.util import tf_run_wrapper
from . import contextmanager
from .random_variable import RandomVariable


def probmodel(builder):
    """
    Decorator to create probabilistic models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not needed that the function returns such variables (we capture
    them using ed.tape).
    """
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        tf.get_default_graph()
        warnings.warn("Provisionally, TF default graph is reset when a prob model is built.")
        return ProbModel(
            builder=builder
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
                model_tape
                self.builder()

            # ed2 RVs created. Relations between them captured in prob_model builder as a networkx graph
            nx_graph = contextmanager.prob_model.get_graph()

            # wrap captured edward2 RVs into inferpy RVs
            model_vars = OrderedDict()
            for k, v in model_tape.items():
                registered_rv = contextmanager.prob_model.get_builder_variable(k)
                if registered_rv is None:
                    # a ed Random Variable. Create a inferpy Random Variable and assign the var directly.
                    model_vars[k] = RandomVariable(v, name=k, is_expanded=False, is_datamodel=False, broadcast_shape=())
                else:
                    model_vars[k] = registered_rv

        return nx_graph, model_vars

    @tf_run_wrapper
    def log_prob(self, sample_dict):
        """ Computes the log probabilities of a (set of) sample(s)"""
        return {k: self.vars[k].log_prob(v) for k, v in sample_dict.items()}

    @tf_run_wrapper
    def sum_log_prob(self, sample_dict):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return tf.reduce_sum([tf.reduce_mean(lp) for lp in self.log_prob(sample_dict).values()])

    @tf_run_wrapper
    def sample(self, size=1):
        """ Generates a sample for eache variable in the model """
        expanded_vars = self._expand_vars(size)
        return {name: tf.convert_to_tensor(var) for name, var in expanded_vars.items()}

    def _expand_vars(self, size):
        """ Create the expanded model vars using sample_shape as plate size and return the OrderedDict """
        with contextmanager.data_model.fit(size=size):
            _, expanded_vars = self._build_model()
        return expanded_vars
