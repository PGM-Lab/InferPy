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
import networkx as nx
from collections import defaultdict, OrderedDict
from tensorflow_probability import edward2 as ed
import tensorflow as tf

from inferpy.util import tf_run_wrapper
from . import contextmanager
from .random_variable import RandomVariable


"""
These set of private functions obtain the dependencies between Random Variables
by analyzing the tensors in Random Varibles and its relations. Finally, the
_get_graph function computes a graph in networkx that represents these dependencies.
"""


def _get_varname(op):
    op_name = op.name
    return op_name[:op_name.find('/')]  # Use the first part of the operation name (until slash) as name


def _children(op):
    # get the consumers of the operation as its children (set of names, using _get_varname)
    return set(_get_varname(opc) for out in op.outputs for opc in out.consumers())


def _clean_graph(G, varnames):
    # G is a networkx graph. Clean nodes from it whose names are not in varnames set or dict.
    # Before removing such nodes, create an edge between their parents and their children.
    g_nodes = list(G.nodes)
    for n in g_nodes:
        if n not in varnames:
            # remove and create edge between parent and child if exist
            for p in G.predecessors(n):
                for s in G.successors(n):
                    G.add_edge(p, s)
            G.remove_node(n)
    return G


def _get_graph(varnames):
    # varnames is a set or dict where keys are the var names of the Random Variables

    # TODO: using default graph to build the model and get the graph.
    # In the future we should use a new graph, or at least allow to give a
    # new one as parameter.

    # Creates dictionary {node: {child1, child2, ..},..} for current
    # TensorFlow graph. Result is compatible with networkx/toposort
    ops = tf.get_default_graph().get_operations()
    dependencies = defaultdict(set)
    for op in ops:
        # in tensorflow_probability, the tensor named *sample_shape* is a op in child-parent order.
        # as we want to capture only the parent-child relations, skip these op names
        # TODO: This is not robust, because sample shape tensors might use a different name (this is the default name)
        if 'sample_shape' not in op.name:
            c = _children(op)
            if len(c) > 0:
                op_name = _get_varname(op)
                c.discard(op_name)  # avoid name references to itself
                dependencies[op_name].update(c)

    # create networkx graph
    G = nx.DiGraph(dependencies)
    # clean names, to get just ed2 RV var names
    _clean_graph(G, varnames)  # inplace modification
    return G


def probmodel(builder):
    """
    Decorator to create probabilistic models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not needed that the function returns such variables (we capture
    them using ed.tape).
    """
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
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
        with contextmanager.prob_model.builder(self):
            # use edward2 model tape to capture RandomVariable declarations
            with ed.tape() as model_tape:
                model_tape
                self.builder()

            # wrap captured edward2 RVs into inferpy RVs
            model_vars = OrderedDict()
            for k, v in model_tape.items():
                registered_rv = contextmanager.prob_model.get_builder_variable(k)
                if registered_rv is None:
                    # a ed Random Variable. Create a inferpy Random Variable and assign the var directly.
                    model_vars[k] = RandomVariable(v, name=k, is_expanded=False, is_datamodel=False, broadcast_shape=())
                else:
                    model_vars[k] = registered_rv

        # ed2 RVs created. Compute the relations between them analyzing the tf computational graph
        nx_graph = _get_graph(model_vars)

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
    def sample(self, sample_shape=()):
        """ Generates a sample for eache variable in the model """
        expanded_vars = self._expand_vars(sample_shape)
        return {name: tf.convert_to_tensor(var) for name, var in expanded_vars.items()}

    def _expand_vars(self, sample_shape):
        """ Create the expanded model vars using sample_shape as plate size and return the OrderedDict """
        with contextmanager.data_model.fit(sample_shape=sample_shape):
            _, expanded_vars = self._build_model()
        return expanded_vars
