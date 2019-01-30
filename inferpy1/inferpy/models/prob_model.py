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
from tensorflow_probability import edward2 as ed

from inferpy.util import tf_run_wrapper
from inferpy.models import RandomVariable


def prob_model(builder):
    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        return ProbModel(
            builder=builder
        )
    return wrapper


class ProbModel:
    def __init__(self, builder):
        self.builder = builder
        with ed.tape() as model_tape:
            self.builder()

        self.vars = {k: RandomVariable(v) for k, v in model_tape.items()}

    @tf_run_wrapper
    def log_prob(self, sample_dict):
        """ Computes the log probabilities of a (set of) sample(s)"""
        return {k: self.vars[k].log_prob(v) for k, v in sample_dict.items()}

    @tf_run_wrapper
    def sum_log_prob(self, sample_dict):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return sum(self.log_prob(sample_dict).values())
