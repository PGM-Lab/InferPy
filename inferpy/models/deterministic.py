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

"""The Deterministic distribution class."""

import edward.models as base_models
import numpy as np
import tensorflow as tf

import inferpy.util
from inferpy.models.random_variable import *
from inferpy.util import get_total_dimension
from inferpy.util import tf_run_wrapper



class Deterministic(RandomVariable):

    """ Class implementing ...


    """

    def __init__(self, loc=None, dim=None, observed=False, name="Determ", ):

        """Construct Deterministic distribution


        """

        if loc != None:

            self.__check_params(loc, dim)   # loc and tensor cannot be None at the same time

            param_dim = 1
            if dim != None: param_dim = dim

            # shape = (batches, dimension)
            self_shape = (inf.replicate.get_total_size(), np.max([get_total_dimension(loc), param_dim]))

            loc_rep = self.__reshape_param(loc, self_shape)

            # build the distribution
            super(Deterministic, self).__init__(base_models.Deterministic(loc=loc_rep, name=name), observed=observed)

        else:

            super(Deterministic, self).__init__(observed=observed)




    @property
    @tf_run_wrapper
    def loc(self):
        """Distribution parameter for the mean."""

        if self.dist != None:
            return self.dist.loc

        return self.base_object



    def __check_params(self, loc, dim):
        """private method that checks the consistency of the input parameters"""


        # loc  cannot be multidimensional arrays (by now)
        if np.ndim(loc) > 1:
            raise ValueError("loccannot be a  multidimensional arrays")


        dim_loc = get_total_dimension(loc)


        # loc can be a scalar or a vector of length dim
        if dim != None and dim_loc > 1 and dim != dim_loc:
            raise ValueError("loc length is not consistent with value in dim")




    def __reshape_param(self,param, self_shape):

        N = self_shape[0]
        D = self_shape[1]


        # get a D*N unidimensional vector

        if np.shape(param) in [(), (1,)] or\
                (isinstance(param, inferpy.models.RandomVariable) and param.dim==1):
            param_vect = np.repeat(param, D * N).tolist()
        else:
            param_vect = np.tile(param, N).tolist()


        if np.all(list(map(lambda x: np.isscalar(x), param_vect))):            # only numerical values

            # reshape the list
            if N > 1:
                param_np_mat = np.reshape(np.stack(param_vect), (N, -1))
            else:
                param_np_mat = np.reshape(np.stack(param_vect), (D,))

            #transform in tf
            param_tf_mat = tf.constant(param_np_mat, dtype="float32")

        else:                                                           # with a tensor

            # transform the numerical values into tensors
            for i in range(0, len(param_vect)):
                if np.isscalar(param_vect[i]):
                    param_vect[i] = [tf.constant(param_vect[i], dtype="float32")]
                elif isinstance(param_vect[i], inferpy.models.RandomVariable):
                    param_vect[i] = param_vect[i].base_object

            # reshape the list
            if N>1:
                param_tf_mat = tf.reshape(tf.stack(param_vect), (N, -1))
            else:
                if D>1:
                    param_tf_mat = tf.reshape(tf.stack(param_vect), (D,))
                else:
                    param_tf_mat = param_vect[0]


        return param_tf_mat


    def __repr__(self):
        return "<inferpy Deterministic '%s', shape=%s>" % (
            self.name, self.shape)



