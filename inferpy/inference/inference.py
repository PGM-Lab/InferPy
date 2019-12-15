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


class Inference:
    """This class implements the functionality of any Inference class.
    """

    def __init__(self):
        raise NotImplementedError

    def compile(self, pmodel, data_size, extra_loss_tensor=None):
        raise NotImplementedError

    def update(self, sample_dict):
        raise NotImplementedError

    def get_interceptable_condition_variables(self):
        # to intercept global and local hidden variables
        return None, None

    def posterior(self, target_names=None, data={}):
        raise NotImplementedError

    def posterior_predictive(self, target_names=None, data={}):
        raise NotImplementedError
