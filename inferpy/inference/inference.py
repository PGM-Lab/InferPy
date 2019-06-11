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


class Inference:
    """This class implements the functionality of any Inference class.
    """

    def __init__(self):
        raise NotImplementedError

    def compile(self, pmodel, data_size):
        raise NotImplementedError

    def update(self, sample_dict):
        raise NotImplementedError

    def sample(self, size=1, data={}):
        raise NotImplementedError

    def log_prob(self, data):
        raise NotImplementedError

    def sum_log_prob(self, data):
        """ Computes the sum of the log probabilities of a (set of) sample(s)"""
        return tf.reduce_sum([tf.reduce_mean(lp) for lp in self.log_prob(data).values()])

    def parameters(names=None):
        raise NotImplementedError
