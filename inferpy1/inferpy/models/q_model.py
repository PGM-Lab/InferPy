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

from . import contextmanager
from .random_variable import RandomVariable


def qmodel(builder):
    """
    Decorator to create q models. The function decorated
    must be a function which declares the Random Variables in the model.
    It is not needed that the function returns such variables (we capture
    them using ed.tape).
    """
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        return QModel(
            builder=lambda: builder(*args, **kwargs)
        )
    return wrapper


class QModel:
    """
    Class that implements the probabilistic model functionality.
    It is composed of a graph, capturing the variable relationships, an OrderedDict containing
    the Random Variables in order of creation, and the function which
    """
    def __init__(self, builder):
        self.builder = builder

    def _build_model(self, pvars):
        tf.reset_default_graph()
        warnings.warn("Provisionally, TF default graph is reset when a q model is compiled and fitted built.")
        # set this graph as active, so datamodel can check and use the model graph
        with contextmanager.q_model.builder(pvars):
            # use edward2 model tape to capture RandomVariable declarations
            with ed.tape() as model_tape:
                self.builder()

            # ed2 RVs created. Relations between them captured in prob_model builder as a networkx graph
            nx_graph = contextmanager.q_model.get_graph()

            # wrap captured edward2 RVs into inferpy RVs
            model_vars = OrderedDict()
            for k, v in model_tape.items():
                registered_rv = contextmanager.q_model.get_builder_variable(k)
                if registered_rv is None:
                    # a ed Random Variable. Create a inferpy Random Variable and assign the var directly.
                    # do not know the args and kwars used to build the ed random variable. Use None.
                    model_vars[k] = RandomVariable(v, name=k, is_expanded=False, var_args=None, var_kwargs=None)
                else:
                    model_vars[k] = registered_rv

        return nx_graph, model_vars
    