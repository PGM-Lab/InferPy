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
import tensorflow as tf
from tensorflow.python.client import session as tf_session

from inferpy import contextmanager
from inferpy import exceptions


class Parameter:
    """
    Random Variable parameter which can be optimized by an inference mechanism.
    """
    def __init__(self, initial_value, name=None):
        self.name = name
        # By defult, parameter is not expanded
        self.is_datamodel = False

        if contextmanager.prob_model.is_active():
            # in this case, the parameter musy have a name
            if self.name is None:
                raise exceptions.NotNamedParameter(
                    'Parameters defined inside a prob model must have a name.')

        # check if Parameter is created inside a prob model and datamodel context or not.
        if contextmanager.prob_model.is_active() and contextmanager.data_model.is_active():
            # In this case, the parameter is in datamodel
            self.is_datamodel = True

            # convert parameter to tensor if it is not
            if not isinstance(initial_value, (tf.Tensor, tf.SparseTensor, tf.Variable)):
                initial_value = tf.convert_to_tensor(initial_value)

            input_varname = initial_value.op.name if contextmanager.prob_model.is_building_graph() else name
            # check the sample_shape. If not empty, expand the initial_value
            contextmanager.prob_model.update_graph(input_varname)

            sample_shape = contextmanager.data_model.get_sample_shape(input_varname)
            if sample_shape is not ():
                initial_value = \
                    tf.broadcast_to(initial_value, tf.TensorShape(sample_shape).concatenate(initial_value.shape))

        # Build the tf variable
        self.var = tf.Variable(initial_value, name=self.name)

        # register the variable in the prob model
        if contextmanager.prob_model.is_active():
            contextmanager.prob_model.register_parameter(self)
            contextmanager.prob_model.update_graph(self.name)


def _tensor_conversion_function(p, dtype=None, name=None, as_ref=False):
    """
        Function that converts the inferpy variable into a Tensor.
        This will enable the use of enable tf.convert_to_tensor(rv)

        If the variable needs to be broadcast_to, do it right now
    """
    return tf.convert_to_tensor(p.var)


# register the conversion function into a tensor
tf.register_tensor_conversion_function(  # enable tf.convert_to_tensor
    Parameter, _tensor_conversion_function)


def _session_run_conversion_fetch_function(p):
    """
        This will enable run and operations with other tensors
    """
    return ([tf.convert_to_tensor(p)], lambda val: val[0])


tf_session.register_session_run_conversion_functions(  # enable sess.run, eval
    Parameter,
    _session_run_conversion_fetch_function)
