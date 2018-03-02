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



from inferpy.util import tf_run_wrapper

from inferpy.prob_model import ProbModel
import tensorflow as tf
import edward as ed






class RandomVariable(object):
    """Base class for random variables.
    """

    def __init__(self, dist, observed):
        self._dist = dist
        self.observed = observed

        if ProbModel.is_active():
            ProbModel.get_active_model().varlist.append(self)


    @property
    def dist(self):
        """Underlying Edward object"""
        return self._dist

    @property
    def dim(self):
        """ Dimensionality of variable """
        return self.dist.shape.as_list()[-1]

    @property
    def batches(self):
        """ Number of batches of the variable"""

        dist_shape = self.dist.shape.as_list()
        if len(dist_shape)>1:
            return dist_shape[-2]
        return 1

    @property
    def shape(self):
        """ shape of the variable, i.e. (batches, dim)"""
        return self.dist.shape.as_list()

    @property
    def name(self):
        """ name of the variable"""
        return self.dist.name[0:-1]


    @property
    def observed(self):
        """ boolean property that determines if a variable is observed or not """
        return self.__observed

    @observed.setter
    def observed(self,observed):
        """ modifies the boolean property that determines if a variable is observed or not """
        self.__observed=observed


    @dist.setter
    def dist(self, dist):
        """ Set the Underlying Edward object"""

        if isinstance(dist, ed.models.RandomVariable)==False:
            raise ValueError("Type of input distribution is nor correct")

        self._dist = dist


    @tf_run_wrapper
    def sample(self, size=1):
        """ Method for obaining a samples

        Args:
            size: scalar or matrix of integers indicating the shape of the matrix of samples.

        Return:
            Matrix of samples. Each element in the output matrix has the same shape than the variable.


        """
        s = self.dist.sample(size)

        return s


    @tf_run_wrapper
    def prob(self, v):
        """ Method for computing the probability of a sample v (or a set of samples)"""
        return self.dist.prob(tf.cast(v, tf.float32))

    @tf_run_wrapper
    def log_prob(self, v):
        """ Method for computing the log probability of a sample v (or a set of samples)"""
        return self.dist.log_prob(tf.cast(v, tf.float32))

    @tf_run_wrapper
    def prod_prob(self, v):
        """ Method for computing the joint probability of a sample v (or a set of samples)"""
        return tf.reduce_prod(self.dist.prob(tf.cast(v, tf.float32)))

    @tf_run_wrapper
    def sum_log_prob(self, v):
        """ Method for computing the sum of the log probability of a sample v (or a set of samples)"""
        return tf.reduce_sum(self.dist.log_prob(tf.cast(v, tf.float32)))

    def __repr__(self):
        return "<inferpy RandomVariable '%s' shape=%s dtype=%s>" % (
            self.name, self.shape, self.dist.dtype.name)



