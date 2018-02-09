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

"""The Normal (Gaussian) distribution class."""

import edward.models as base_models
import numpy as np
import inferpy.util
from inferpy.replicate import *


# \frac{1}{{\sigma \sqrt {2\pi } }}e^{{{ - \left( {x - \mu } \right)^2 }\mathord{\left/ {\vphantom {{ - \left( {x - \mu } \right)^2 } {2\sigma ^2 }}} \right. \kern-\nulldelimiterspace} {2\sigma ^2 }}}

#..math::
#
#f(x |\mu, \sigma ^ 2)=
#
#         e^{{{ - \left( {x - \mu } \right)^2 }\mathord{\left/ {\vphantom {{ - \left( {x - \mu } \right)^2 } {2\sigma ^2 }}} \right.\kern-\nulldelimiterspace} {2\sigma ^2 }}}



class Normal:

    """ Class implementing the Normal distribution with location `loc`, `scale` and `dim` parameters.

    The probability density of the normal distribution is,

    .. math::

      f(x|\mu,\sigma^2)=\\frac{1}{{\\sigma \\sqrt {2\\pi}}} e^{-\\frac{(x-\\mu)^2}{2 \\sigma ^2}}


    where

    - :math:`\mu`  is the mean or expectation of the distribution (i.e. `location`),
    - :math:`\sigma`  is the standard deviation (i.e. `scale`), and
    - :math:`\sigma^{2}` is the variance.



    The Normal distribution is a member of the `location-scale
    family <https://en.wikipedia.org/wiki/Location-scale_family>`_.

    This class allows the definition of a variable normal distributed of
    any dimension. Each of the dimensions are independent. For example:

    .. literalinclude:: ../../examples/normal_dist_definition.py



    """

    def __init__(self, loc, scale, dim=None, name="inf_Normal"):

        """Construct Normal distributions

        The parameters `loc` and `scale` must be shaped in a way that supports
        broadcasting (e.g. `loc + scale` is a valid operation). If dim is specified,
        it should be consistent with the lengths of `loc` and `scale`


        Args:
            loc (float): scalar or vector indicating the mean of the distribution at each dimension.
            scale (float): scalar or vector indicating the stddev of the distribution at each dimension.
            dim (int): optional scalar indicating the number of dimensions

        Raises
            ValueError: if the parameters are not consistent
            AttributeError: if any of the properties is changed once the object is constructed

        """


        self.__check_params(loc, scale, dim)


        param_dim = 1
        if dim != None: param_dim

        # shape = (batches, dimension)
        self_shape = (replicate.get_total_size(), np.max([np.size(loc), np.size(scale), param_dim]))

        # build the loc and scale matrix
        if np.isscalar(loc):
            loc_rep = np.tile(loc, (self_shape[0], self_shape[1]))
        else:
            loc_rep = np.tile(loc, (self_shape[0], 1))

        if np.isscalar(scale):
            scale_rep = np.tile(scale, (self_shape[0], self_shape[1]))
        else:
            scale_rep = np.tile(scale, (self_shape[0], 1))

        # build the distribution

        self.__dist = base_models.Normal(loc=loc_rep, scale=scale_rep, name=name)

    # getter methods

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return inferpy.util.runtime.tf_sess.run(self.dist.loc)

    @property
    def scale(self):
        """Distribution parameter for standard deviation."""
        return inferpy.util.runtime.tf_sess.run(self.dist.scale)

    @property
    def dim(self):
        """ Dimensionality of variable """
        return self.dist.shape.as_list()[1]

    @property
    def batches(self):
        """ Number of batches of the variable"""
        return self.dist.shape.as_list()[0]

    @property
    def shape(self):
        """ shape of the variable, i.e. (batches, dim)"""
        return self.dist.shape.as_list()

    @property
    def dist(self):
        """Underlying Edward object"""
        return self.__dist



    def __check_params(self, loc, scale, dim):
        """private method that checks the consistency of the input parameters"""

        # loc and scale cannot be multidimensional arrays (by now)
        if np.ndim(loc) > 1 or np.ndim(scale) > 1:
            raise ValueError("loc and scale cannot be multidimensional arrays")

        len_loc = np.size(loc)
        len_scale = np.size(scale)

        # loc and scale lengths must be equal or must be scalars
        if len_loc > 1 and len_scale > 1 and len_loc != len_scale:
            raise ValueError("loc and scale lengths must be equal or must be 1")

        # loc can be a scalar or a vector of length dim

        if dim != None and len_loc > 1 and dim != len_loc:
            raise ValueError("loc length is not consistent with value in dim")

        if dim != None and len_scale > 1 and dim != len_scale:
            raise ValueError("scale length is not consistent with value in dim")


    def sample(self, v):
        """ Method for obaining a sample of shape v"""
        return inferpy.util.runtime.tf_sess.run(self.dist.sample(v))
