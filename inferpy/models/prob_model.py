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
import warnings

from inferpy import util
from inferpy import contextmanager
from inferpy.queries import Query
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
    the Random Variables/Parameters in order of creation, and the function which declare the
    Random Variables/Parameters.
    """

    def __init__(self, builder):
        # Initialize object attributes
        self.builder = builder
        g_for_nxgraph = tf.Graph()
        # first buid the graph of dependencies
        with g_for_nxgraph.as_default():
            with tf.Session() as sess:
                try:
                    default_sess = util.session.swap_session(sess)
                    self.graph = self._build_graph()
                finally:
                    # sess is closed by context, no need to use set_session (which closes the actual running session)
                    util.session.swap_session(default_sess)

        # Now initialize vars and params for the model (no sample_shape)
        self.vars, self.params = self._build_model()

        # This attribute contains the inference method used. If it is None, the `fit` function has not been used yet
        self.inference_method = None

        self.observed_vars = []  # list of variables that have been observed during the inference

    # all the results of prior, posterior and posterior_predictive are evaluated always, because they depends on
    # tf.Variables, and therefore a tensor cannot be return because the results would depend on the value of that
    # tf.Variables

    def prior(self, target_names=None, data={}):
        return Query(self.vars, target_names, data)

    def posterior(self, target_names=None, data={}):
        if self.inference_method is None:
            raise RuntimeError("posterior cannot be used before using the fit function.")

        # only non-observed variables can be in target_names
        if target_names is None:
            target_names = [name for name in self.vars.keys() if name not in self.observed_vars]
        else:
            if any(var in self.observed_vars for var in target_names):
                raise ValueError("target_names must correspond to not observed variables during the inference: \
                    {}".format([v for v in self.vars.keys() if v not in self.observed_vars]))

        prior_data = self._create_hidden_observations(target_names, data)

        return Query(self.inference_method.expanded_variables["q"], target_names, {**data, **prior_data})

    def posterior_predictive(self, target_names=None, data={}):
        if self.inference_method is None:
            raise RuntimeError("posterior_preductive cannot be used before using the fit function.")

        # only non-observed variables can be in target_names
        if target_names is None:
            target_names = [name for name in self.vars.keys() if name in self.observed_vars]
        else:
            if any(var not in self.observed_vars for var in target_names):
                raise ValueError("target_names must correspond to observed variables during the inference: \
                    {}".format(self.observed_vars))

        prior_data = self._create_hidden_observations(target_names, data)

        return Query(self.inference_method.expanded_variables["p"], target_names, {**data, **prior_data})

    def _create_hidden_observations(self, target_names, data={}):
        # TODO: This code must be implemented independent of the inference method. Right now we are using the p and q
        # expanded variables, which belongs only to variational inference methods. When a different VI is implemented
        # think about a better way to implement this function and access to the correct dict of random variables

        # NOTE: implementation trick. As p model variables are intercepted with q model variables,
        # compute prior observations for local hidden variables which are not targets,
        # expanding a new model using plate_size and then sampling
        hidden_variable_names = [k for k in self.vars.keys() if k not in target_names and k not in data]
        if hidden_variable_names:
            expanded_vars, _ = self.expand_model(self.inference_method.plate_size)
            prior_data = Query(expanded_vars, hidden_variable_names, data).sample(simplify_result=False)
        else:
            prior_data = {}

        return prior_data

    def _build_graph(self):
        with contextmanager.randvar_registry.init():
            self.builder()
            # ed2 RVs created. Relations between them captured in randvar_registry builder as a networkx graph
            nx_graph = contextmanager.randvar_registry.get_graph()

        return nx_graph

    def _build_model(self):
        # get the global variables defined before building the model
        _before_global_variables = tf.global_variables()

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

        # get the global variables defined after building the model
        _after_global_variables = tf.global_variables()
        # compute the new global variables defined when building the model
        created_vars = [v for v in _after_global_variables if v not in _before_global_variables]
        util.get_session().run(tf.variables_initializer(created_vars))

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

        # if fit was called before, warn that it restarts everything
        if self.inference_method:
            warnings.warn("Fit was called before. This will restart the inference method and \
                re-build the expanded model.")

        # set the inference method
        self.inference_method = inference_method

        # get the plate size
        plate_size = util.iterables.get_plate_size(self.vars, sample_dict)
        # compile the inference method
        inference_method.compile(self, plate_size)
        # and run the update method with the data
        inference_method.update(sample_dict)

        # If it works, set the observed variables
        self.observed_vars = list(sample_dict.keys())

    @util.tf_run_ignored
    def update(self, sample_dict):
        # Check that fit was called first
        if self.inference_method is None:
            raise RuntimeError("The fit method must be called before update")

        # check that the observed_vars are the same
        if set(self.observed_vars) != sample_dict.keys():
            raise ValueError("The data in sample dict must contain only data from observed variables: \
                {}.".format(self.observed_vars))

        # Run the inference method
        self.inference_method.update(sample_dict)

    def expand_model(self, size=1):
        """ Create the expanded model vars using size as plate size and return the OrderedDict """

        with contextmanager.data_model.fit(size=size):
            expanded_vars, expanded_params = self._build_model()

        return expanded_vars, expanded_params
