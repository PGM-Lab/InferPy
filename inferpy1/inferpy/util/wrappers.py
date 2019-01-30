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
Module with useful wrappers used for the development of InferPy.
"""

from functools import wraps

from inferpy import util


def tf_run_wrapper(f):
    """ 
    A function might return a tensor or not. In order to decide if the result of this
    function needs to be evaluated in a tf session or not, wrap the output using the
    tf_run_eval function from the utils.Runtime module.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "tf_run" in kwargs:
            tf_run = kwargs.pop("tf_run")
        else:
            tf_run = util.tf_run_default
        
        # Just run tensors in tf session, if required, in the outter level.
        # This way, we do not care about calls inside decorated functions (tensors are never evaluated)
        # First, disable tf_run_default to avoid inner decorated functions to be evaluated using the default value
        _tmp = util.tf_run_default
        util.tf_run_default = False
        # eval the function
        result = util.tf_run_eval(f(*args, **kwargs), tf_run=tf_run)
        # restore the default tf_run
        util.tf_run_default = _tmp
        # return the result
        return result
    return wrapper
