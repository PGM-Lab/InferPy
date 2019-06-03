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
Module focused on evaluating tensors to makes the usage easier, forgetting about tensors and sessions
"""

from functools import wraps
from contextlib import contextmanager

from inferpy import util


# default value for tf_run in decorated tf_run_allowed functions
__tf_run_default = True

# configuration environment for runner_scopes. It counts the number of nested contexts
runner_context = dict(
    runner_recursive_depth=0
)


@contextmanager
def runner_scope():
    # Update the runner recursive depth, because decorated functions might call other decorated functions too.
    # This way, we can control that only first level decorated functions will be evaluated in a tf Session.
    runner_context['runner_recursive_depth'] += 1
    try:
        yield
    finally:
        runner_context['runner_recursive_depth'] -= 1


def tf_run_allowed(f):
    """
    A function might return a tensor or not. In order to decide if the result of this function needs to be evaluated
    in a tf session or not, use the tf_run extra parameter or the tf_run_default value. If True, and this function is
    in the first level of execution depth, use a tf Session to evaluate the tensor or other evaluable object (like dicts)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # first obtains the tf_run, a bool which tells if we need to eval the output in a session or not
        if "tf_run" in kwargs:
            tf_run = kwargs.pop("tf_run")
        else:
            tf_run = __tf_run_default

        # use this context to keep track of the decorated functions calls (recursive depth level)
        with runner_scope():
            # now execute the function
            obj = f(*args, **kwargs)
            if tf_run and runner_context['runner_recursive_depth'] == 1:
                # first recursive depth, and tf_run is True: we can eval the function
                return try_run(obj)
            else:
                # tf_run is False or we are in a deeper runner levels than 1 (do not eval the result yet)
                return obj
    return wrapper


def tf_run_ignored(f):
    """
    A function might call other functions decorated with tf_run_allowed.
    This decorator is used to avoid that such functions are evaluated.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # use this context to keep track of the decorated functions calls (recursive depth level).
        # this way, deeper execution level will not be evaluated.
        # finally, in first execution level we do not evaluate the result neither
        with runner_scope():
            # now execute the function and return the result
            return f(*args, **kwargs)

    return wrapper


def set_tf_run(enable):
    # this function is used to modify the default state of tf run (eval tensors or not)
    global __tf_run_default
    __tf_run_default = enable


def try_run(obj):
    try:
        ev_obj = util.get_session().run(obj)
        return ev_obj
    except (RuntimeError, TypeError, ValueError):
        # cannot evaluate the result, return the obj
        return obj
