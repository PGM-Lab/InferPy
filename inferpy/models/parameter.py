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
from inferpy import util


class Parameter:
    """
    Random Variable parameter which can be optimized by an inference mechanism.
    """
    def __init__(self, initial_value, name=None):
        # By defult, parameter is not expanded
        self.is_datamodel = False

        # the parameter must have a name
        self.name = name if name else util.name.generate('parameter')

        # check if Parameter is created inside a datamodel context or not.
        if contextmanager.data_model.is_active():
            # In this case, the parameter is in datamodel
            self.is_datamodel = True

            # convert parameter to tensor if it is not
            initial_value = tf.cast(initial_value, tf.float32)

            input_varname = initial_value.op.name if contextmanager.randvar_registry.is_building_graph() else name
            # check the sample_shape. If not empty, expand the initial_value
            contextmanager.randvar_registry.update_graph(input_varname)

            sample_shape = contextmanager.data_model.get_sample_shape(input_varname)
            if sample_shape is not ():
                initial_value = \
                    tf.broadcast_to(initial_value, tf.TensorShape(sample_shape).concatenate(initial_value.shape))

        # Build the tf variable
        self.var = tf.Variable(initial_value, name=self.name)
        util.session.get_session().run(tf.variables_initializer([self.var]))

        # register the variable, which is used to detect dependencies
        contextmanager.randvar_registry.register_parameter(self)
        contextmanager.randvar_registry.update_graph()


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
