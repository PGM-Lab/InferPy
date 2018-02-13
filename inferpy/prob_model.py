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

class ProbModel():
    """Class implementing a probabilistic model


        """

    __active_models = []

    def __init__(self,distlist=[]):
        """Initializes the ...

        Args:
            ...

        """

        for d in distlist:
            if not isinstance(d, inferpy.models.RandomVariable):
                raise ValueError("The input argument is not a list of RandomVariables")

        if ProbModel.is_active():
            raise inferpy.util.ScopeException("Nested probabilistic models cannot be defined")


        self.distlist=distlist

    # properties and setters

    @property
    def distlist(self):
        return self.__distlist

    @distlist.setter
    def distlist(self,distlist):
        self.__distlist = distlist


    # other methods

    def compile(self):
        pass

    def __enter__(self):
        ProbModel.__active_models.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ProbModel.__active_models.pop()

    def add_dist(self, d):

        if isinstance(d, inferpy.models.RandomVariable) == False:
            raise ValueError("The input argument is not a RandomVariable")

        if(d not in self.distlist):
            self.distlist.append(d)

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



