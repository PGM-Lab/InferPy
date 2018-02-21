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


"""Module with the probabilistic model functionality.
"""

import inferpy.util
import inferpy.models
from inferpy.util import tf_run_wrapper


class ProbModel():
    """Class implementing a probabilistic model


        """

    __active_models = []

    def __init__(self, varlist=[]):
        """Initializes the ...

        Args:
            ...

        """

        for d in varlist:
            if not isinstance(d, inferpy.models.RandomVariable):
                raise ValueError("The input argument is not a list of RandomVariables")

        if ProbModel.is_active():
            raise inferpy.util.ScopeException("Nested probabilistic models cannot be defined")


        self.varlist=varlist

    # properties and setters

    @property
    def varlist(self):
        return self.__varlist

    @varlist.setter
    def varlist(self,varlist):
        self.__varlist = varlist


    # other methods

    def compile(self):
        pass

    def get_observed_vars(self):
        vl = []
        for v in self.varlist:
            if v.observed:
                vl.append(v)
        return vl

    def get_latent_vars(self):
        vl = []
        for v in self.varlist:
            if v.observed==False:
                vl.append(v)
        return vl






    def __enter__(self):
        ProbModel.__active_models.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ProbModel.__active_models.pop()

    def add_var(self, v):

        if isinstance(v, inferpy.models.RandomVariable) == False:
            raise ValueError("The input argument is not a RandomVariable")

        if v not in self.varlist:
            self.varlist.append(v)


    def get_var(self,name):
        for v in self.varlist:
            if v.name ==  name:
                return v
        return None



    @tf_run_wrapper
    def log_prob(self, sample_dict):

        sd = {}

        for k, v in sample_dict.iteritems():
            var=self.get_var(k)
            sd.update({k: var.log_prob(v, tf_run=False)})


        return sd

    @tf_run_wrapper
    def sum_log_prob(self, sample_dict):

        lp=0

        for k, v in sample_dict.iteritems():
            var = self.get_var(k)
            lp += var.sum_log_prob(v, tf_run=False)

        return lp

    @tf_run_wrapper
    def sample(self, size=1):
        sd = {}
        for v in self.varlist:
            sd.update({v.name:v.sample(size, tf_run=False)})

        return sd

    def get_config(self):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


    # static methods
    @staticmethod
    def get_active_model():
        if ProbModel.is_active():
            return ProbModel.__active_models[-1]
        return []

    @staticmethod
    def is_active():
        """Check if a replicate construct has been initialized

        Returns:
             True if the method is inside a construct ProbModel (of size different to 1).
             Otherwise False is return
        """
        return len(ProbModel.__active_models)>0



