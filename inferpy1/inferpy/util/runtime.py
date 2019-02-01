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


"""
Module with useful definitions to be used in runtime
"""

import tensorflow as tf

from inferpy import util

# configuration for the tf evaluation in sessions
tf_run_default = True


def tf_run_eval(obj, tf_run=None):
    """
    Check if the obj object needs to be evaluated in a tf session or not.
    :param obj: Object to test if it needs to be evaluated in a tf session or not
    :param tf_run: Check if eval the `obj` or not. If None, use the `inferpy.util.Runtime.tf_run_default`,
    otherwise is a boolean telling if evaluate `obj` or not.
    """
    # If None, use the default tf_run declared in util
    run_sess = tf_run
    if run_sess is None:
        run_sess = util.tf_run_default

    # if it is a function, wrap it such that the return obj pass through this function
    if hasattr(obj, '__call__'):
        return util.tf_run_wrapper(obj)
    # else, if not run_sess, return the obj as it is
    elif not run_sess:
        return obj
    # otherwise, try to evaluate the obj in a session. It can be a Tensor or not, an iterable with Tensors or not.
    # If it fails to run the obj in a session, return the object as it is. Otherwise return the evaluated obj.
    else:
        try:
            with tf.Session() as sess:
                # Run variable initializers
                __init_g = tf.global_variables_initializer()
                __init_l = tf.local_variables_initializer()
                sess.run([__init_g, __init_l])
                # Run obj in sess
                result = sess.run(obj)
            return result
        except Exception:
            return obj