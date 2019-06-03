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


from enum import IntEnum
import functools
from collections import OrderedDict
from tensorflow_probability import edward2 as ed
import tensorflow as tf
import networkx as nx
from matplotlib import pyplot as plt

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

    # all the results of prior, posterior and posterior_predictive are evaluated always, because they depends on
    # tf.Variables, and therefore a tensor cannot be return because the results would depend on the value of that
    # tf.Variables

    def prior(self, target_names, data={}):
        return Query(self.vars, target_names, data)

    def posterior(self, target_names, data={}):
        if self.inference_method is None:
            raise RuntimeError("posterior cannot be used before using the fit function.")

        prior_data = self._create_hidden_observations(target_names, data)

        return Query(self.inference_method.expanded_variables["q"], target_names, {**data, **prior_data})

    def posterior_predictive(self, target_names, data={}):
        if self.inference_method is None:
            raise RuntimeError("posterior_preductive cannot be used before using the fit function.")

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

        # set the inference method
        self.inference_method = inference_method

        # Run the inference method
        inference_method.run(self, sample_dict)

    @util.tf_run_allowed
    def log_prob(self, data):
        """ Computes the log probabilities of a (set of) sample(s)"""
        with contextmanager.observe(self.vars, data):
            return {k: self.vars[k].log_prob(tf.convert_to_tensor(v)) for k, v in data.items()}

    @util.tf_run_allowed
    def sum_log_prob(self, data):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return tf.reduce_sum([tf.reduce_mean(lp) for lp in self.log_prob(data).values()])

    @util.tf_run_allowed
    def sample(self, size=1, data={}):
        """ Generates a sample for eache variable in the model """
        expanded_vars, expanded_params = self.expand_model(size)
        with ed.interception(util.interceptor.set_values(**data)):
            expanded_vars, expanded_params = self.expand_model(size)
        return {name: tf.convert_to_tensor(var) for name, var in expanded_vars.items()}

    @util.tf_run_allowed
    def parameters(self, names=None):
        """ Return the parameters of the Random Variables of the model.
        If `names` is None, then return all the parameters of all the Random Variables.
        If `names` is a list, then return the parameters specified in the list (if exists) for all the Random Variables.
        If `names` is a dict, then return all the parameters specified (value) for each Random Variable (key).

        NOTE: If tf_run=True, but any of the returned parameters is not a Tensor *and therefore cannot be evaluated)
            this returns a not evaluated dict (because the evaluation will raise an Exception)

        Args:
            names: A list, a dict or None. Specify the parameters for the Random Variables to be obtained.

        Returns:
            A dict, where the keys are the names of the Random Variables and the values a dict of parameters (name-value)
        """
        # argument type checking
        if not(names is None or isinstance(names, (list, dict))):
            raise TypeError("The argument 'names' must be None, a list or a dict, not {}.".format(type(names)))
        # now we can assume that names is None, a list or a dict

        # function to filter the parameters for each Random Variable
        def filter_parameters(varname, parametes):
            if names is None:
                return parametes

            selected_parameters = set(names if isinstance(names, list) else names[varname])

            return {k: v for k, v in parametes.items() if k in selected_parameters}

        return {k: filter_parameters(v.name, v.parameters) for k, v in self.vars.items()
                # filter variables based on names attribute
                if names is None or isinstance(names, list) or k in names}

    def expand_model(self, size=1):
        """ Create the expanded model vars using size as plate size and return the OrderedDict """

        with contextmanager.data_model.fit(size=size):
            expanded_vars, expanded_params = self._build_model()

        return expanded_vars, expanded_params
