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
import warnings

from inferpy import util
from inferpy import contextmanager
from inferpy.queries import Query
from .random_variable import RandomVariable
from inferpy.data.loaders import build_data_loader


def probmodel(builder):
    """
    Decorator to create probabilistic models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not required that the function returns such variables (they are
    captured using ed.tape).
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

        # losses tensor if model contains any inferpy layers Sequential object
        self.layer_losses = None

    # all the results of prior, posterior and posterior_predictive are evaluated always, because they depends on
    # tf.Variables, and therefore a tensor cannot be return because the results would depend on the value of that
    # tf.Variables

    def prior(self, target_names=None, data={}, size_datamodel=1):


        if size_datamodel > 1:
            variables, _ = self.expand_model(size_datamodel)
        elif size_datamodel == 1:
            variables = self.vars
        else:
            raise ValueError("size_datamodel must be greater than 0 but it is {}".format(size_datamodel))

        util.init_uninit_vars()

        return Query(variables, target_names, data)

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

        return self.inference_method.posterior(target_names, data)

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

        return self.inference_method.posterior_predictive(target_names, data)

    def _build_graph(self):
        with contextmanager.randvar_registry.init():
            self.builder()
            # ed2 RVs created. Relations between them captured in randvar_registry builder as a networkx graph
            nx_graph = contextmanager.randvar_registry.get_graph()

        return nx_graph

    def _build_model(self):

        with contextmanager.randvar_registry.init(self.graph):
            with contextmanager.layer_registry.init():
                # use edward2 model tape to capture RandomVariable declarations
                with ed.tape() as model_tape:
                    self.builder()

                # store the losses from the build layers through layers.sequential.Sequential
                # NOTE: this must be done inside the layer_registry context, where the sequentials are stored
                self.layer_losses = contextmanager.layer_registry.get_losses()

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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("The function plot_graph requires to install inferpy[visualization]")
            raise
        nx.draw(self.graph, cmap=plt.get_cmap('jet'), with_labels=True)
        plt.show()

    @util.tf_run_ignored
    def fit(self, data, inference_method):
        # Parameter checkings
        # sample_dict must be a non empty python dict or dataloader
        data_loader = build_data_loader(data)
        plate_size = data_loader.size

        if len(data_loader.variables) == 0:
            raise ValueError('The number of mapped variables must be at least 1.')

        # if fit was called before, warn that it restarts everything
        if self.inference_method:
            warnings.warn("Fit was called before. This will restart the inference method and \
                re-build the expanded model.")

        # set the inference method
        self.inference_method = inference_method

        # compile the inference method
        # if the inference method needs to intercept random variables, enable this context using a boolean
        # tf.Variable defined in this inference method object
        with util.interceptor.enable_interceptor(*self.inference_method.get_interceptable_condition_variables()):
            inference_method.compile(self, plate_size, self.layer_losses)
            # and run the update method with the data
            inference_method.update(data_loader)

        # If it works, set the observed variables
        self.observed_vars = data_loader.variables

    def expand_model(self, size=1):
        """ Create the expanded model vars using size as plate size and return the OrderedDict """

        with contextmanager.data_model.fit(size=size):
            expanded_vars, expanded_params = self._build_model()

        return expanded_vars, expanded_params
