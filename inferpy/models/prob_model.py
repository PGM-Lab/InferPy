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
import networkx as nx
from matplotlib import pyplot as plt

from inferpy import util
from inferpy import exceptions
from inferpy import contextmanager
from .random_variable import RandomVariable


# global variable to know if the prob model is being built or not
is_probmodel_building = False


def build_model(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global is_probmodel_building
        try:
            is_probmodel_building = True
            return f(*args, **kwargs)
        finally:
            is_probmodel_building = False
    return wrapper


def probmodel(builder):
    """
    Decorator to create probabilistic models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not needed that the function returns such variables (we capture
    them using ed.tape).
    """
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        @util.tf_run_ignored
        def fn():
            return builder(*args, **kwargs)
        return ProbModel(
            builder=fn
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
        self.builder = builder
        g_for_nxgraph = tf.Graph()
        with g_for_nxgraph.as_default():
            self.graph = self._build_graph()
        # Now initialize vars and params for the model (no sample_shape)
        self.vars, self.params = self._build_model()
        self._last_expanded_vars = None
        self._last_expanded_params = None
        self._last_fitted_vars = None
        self._last_fitted_params = None

    @property
    def posterior(self):
        if self._last_fitted_vars is None:
            raise RuntimeError("posterior cannot be accessed before using the fit function.")
        return self._last_fitted_vars

    @build_model
    def _build_graph(self):
        with contextmanager.randvar_registry.init():
            self.builder()
            # ed2 RVs created. Relations between them captured in randvar_registry builder as a networkx graph
            nx_graph = contextmanager.randvar_registry.get_graph()

        return nx_graph

    @build_model
    def _build_model(self):
        with contextmanager.randvar_registry.init(self.graph):
            # use edward2 model tape to capture RandomVariable declarations
            with ed.tape() as model_tape:
                self.builder()

            # get variables from parameters
            var_parameters = contextmanager.randvar_registry.get_var_parameters()

            # wrap captured edward2 RVs into inferpy RVs
            model_vars = OrderedDict()
            for k, v in model_tape.items():
                registered_rv = contextmanager.randvar_registry.get_variable(k)
                if registered_rv is None:
                    # a ed Random Variable. Create a inferpy Random Variable and assign the var directly.
                    # do not know the args and kwars used to build the ed random variable. Use None.
                    model_vars[k] = RandomVariable(v, name=k, is_datamodel=False, ed_cls=None,
                                                   var_args=None, var_kwargs=None, sample_shape=())
                else:
                    model_vars[k] = registered_rv

        return model_vars, var_parameters

    def plot_graph(self):
        nx.draw(self.graph, cmap=plt.get_cmap('jet'), with_labels=True)
        plt.show()

    @util.tf_run_ignored
    def fit(self, sample_dict, inference_method):
        # Parameter checkings
        # sample_dict must be a non empty python dict
        if not isinstance(sample_dict, dict):
            raise TypeError('The `sample_dict` type must be dict.')
        if len(sample_dict) == 0:
            raise ValueError('The number of mapped variables must be at least 1.')

        # Run the inference method
        fitted_vars, fitted_params = inference_method.run(self, sample_dict)
        self._last_fitted_vars = fitted_vars
        self._last_fitted_params = fitted_params

        return self.posterior

    @util.tf_run_allowed
    def log_prob(self, sample_dict):
        """ Computes the log probabilities of a (set of) sample(s)"""
        return {k: self.vars[k].log_prob(v) for k, v in sample_dict.items()}

    @util.tf_run_allowed
    def sum_log_prob(self, sample_dict):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return tf.reduce_sum([tf.reduce_mean(lp) for lp in self.log_prob(sample_dict).values()])

    @util.tf_run_allowed
    def sample(self, size=1):
        """ Generates a sample for eache variable in the model """
        expanded_vars, expanded_params = self.expand_model(size)
        return {name: tf.convert_to_tensor(var) for name, var in expanded_vars.items()}

    def expand_model(self, size=None):
        """ Create the expanded model vars using size as plate size and return the OrderedDict """

        with contextmanager.data_model.fit(size=size):
            expanded_vars, expanded_params = self._build_model()

        self._last_expanded_vars = expanded_vars
        self._last_expanded_params = expanded_params

        return expanded_vars, expanded_params

    def _get_plate_size(self, sample_dict):
        # get the plate size by analyzing the sample_dict input
        # check that all values in dict whose name is a datamodel RV has the same length (will be the plate size)
        plate_shapes = [util.iterables.get_shape(v) for k, v in sample_dict.items()
                        if k in self.vars and self.vars[k].is_datamodel]
        plate_sizes = [s[0] if len(s) > 0 else 1 for s in plate_shapes]  # if the shape is (), it is just one element
        if len(plate_sizes) == 0:
            return 1
        else:
            plate_size = plate_sizes[0]

            if any(plate_size != x for x in plate_sizes[1:]):
                raise exceptions.InvalidParameterDimension(
                    'The number of elements for each mapped variable must be the same.')

            return plate_size
