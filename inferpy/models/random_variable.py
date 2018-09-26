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


""" Module implementing the shared functionality across all the variable types  """


from inferpy.util import tf_run_wrapper
from inferpy.util import static_multishape
import inferpy as inf
import tensorflow as tf
import edward as ed
import numpy as np
import sys
import collections





class RandomVariable(object):
    """Base class for random variables.
    """


    __declared_vars = {}

    def __init__(self, base_object=None, observed=False):

        """ Constructor for the RandomVariable class

            Args:
                base_object: encapsulated Edward object (optional).
                observed (bool): specifies if the random variable is observed (True) or observed (False).



        """


        self.base_object = base_object
        self.__bind = None
        self.observed = observed
        self.copied_from = None

        if base_object != None:



            if  inf.ProbModel.is_active() and not self.is_generic_variable():
                inf.ProbModel.get_active_model().varlist.append(self)

            if inf.replicate.in_replicate():
                for r in inf.replicate.get_active_replicate():
                    r.varlist.append(self)

        RandomVariable.__declared_vars.update({id(self) : self})



    @property
    def dist(self):
        """Underlying Edward object"""

        if self.is_generic_variable():
            return None

        return self._base_object

    @property
    def base_object(self):
        """Underlying Tensorflow object"""

        return self._base_object



    @property
    def dim(self):
        """ Dimensionality of variable """
        return self.base_object.shape.as_list()[- (1 + len(self.event_shape)) ]

    @property
    def batches(self):
        """ Number of batches of the variable"""

        dist_shape = self.base_object.shape.as_list()

        if len(dist_shape) - len(self.event_shape) <= 1 :
            return 1
        return dist_shape[0]

    @property
    def shape(self):
        """ shape of the variable, i.e. (batches, dim)"""
        return self.base_object.shape.as_list()

    @property
    def event_shape(self):
        """ event_shape"""
        if self.is_generic_variable():
            return []

        return self.base_object.event_shape.as_list()


    @property
    def name(self):
        """ name of the variable"""
        return self.base_object.name[0:-1]


    @property
    def observed(self):
        """ boolean property that determines if a variable is observed or not """
        return self.__observed

    @observed.setter
    def observed(self,observed):
        """ modifies the boolean property that determines if a variable is observed or not """
        self.__observed=observed


    @property
    def bind(self):
        """  """
        return self.__bind

    @bind.setter
    def bind(self,bind):
        """  """

        if not isinstance(bind, RandomVariable):
            raise ValueError("object to bind is not RandomVariable")

        self.__bind=bind



    @dist.setter
    def dist(self, dist):
        """ Set the Underlying Edward object"""

        if isinstance(dist, ed.models.RandomVariable)==False:
            raise ValueError("Type of input distribution is not correct")

        self._base_object = dist

    @base_object.setter
    def base_object(self, tensor):
        """ Set the Underlying tensorflow object"""

        if isinstance(tensor, tf.Tensor)==False and \
                        isinstance(tensor, ed.RandomVariable) == False and tensor != None:
            raise ValueError("Type of input object is not correct")

        self._base_object = tensor


    @tf_run_wrapper
    def sample(self, size=1):
        """ Method for obaining a samples

        Args:
            size: scalar or matrix of integers indicating the shape of the matrix of samples.

        Return:
            Matrix of samples. Each element in the output matrix has the same shape than the variable.


        """

        if self.is_generic_variable():
            return self.base_object


        s = self.dist.sample(size)

        if self.dim > 1 and self.batches == 1:


            final_shape = (size, self.dim) if self.event_shape == [] else (size, self.dim, self.event_shape[0])

            s = tf.reshape(s, shape=final_shape)

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

    @tf_run_wrapper
    def mean(self, name='mean'):

        """ Method for obaining the mean of this random variable """

        if not self.is_generic_variable():
            return self.dist.mean(name)
        return None

    @tf_run_wrapper
    def variance(self, name='variance'):

        """ Method for obataining the variance of this random variable"""

        if not self.is_generic_variable():
            return self.dist.variance(name)
        return None

    @tf_run_wrapper
    def stddev(self, name='stddev'):

        """ Method for obataining the standard deviation of this random variable"""


        if not self.is_generic_variable():
            return self.dist.stddev(name)
        return None

    def is_generic_variable(self):
        """ Determines if this is a generic variable, i.e., an Edward variable
        is not encapsulated. """

        return isinstance(self._base_object, ed.RandomVariable) == False


    def get_replicate_list(self):

        """ Returns a list with all the replicate constructs that this variable belongs to.  """

        return [r  for r in inf.replicate.get_all_replicate() if self in r.varlist]


    def get_local_hidden(self):

        """ Returns a list with all the local hidden variables w.r.t. this one. Local hidden variables
        are those latent variables which are in the same replicate construct.  """

        var_rep = [r.varlist for r in self.get_replicate_list()]

        var_rep_id = [inf.models.RandomVariable.get_key_from_var(vr) for vr in var_rep]

        if len(var_rep) == 0:
            intersect_id = []
        elif len(var_rep) == 1:
            intersect_id = var_rep_id[0]
        else:

            for i in range(0, len(var_rep_id)):
                if i == 0:
                    intersect_id = var_rep_id[0]
                else:
                    intersect_id = np.intersect1d(intersect_id, var_rep_id[i])

        intersect_id = [x for x in intersect_id if inf.models.RandomVariable.get_key_from_var(self) != x]
        intersect = inf.models.RandomVariable.get_var_with_key(intersect_id)
        intersect = [x for x in intersect if not x.observed]

        return intersect

    def __repr__(self):

        """ Representation for this variable"""

        str = "<inferpy RandomVariable '%s' shape=%s dtype=%s>" % (
            self.name, self.shape, self.base_object.dtype.name)

        return str


    @staticmethod
    @static_multishape
    def get_var_with_key(key):
        return RandomVariable.__declared_vars.get(key)

    @staticmethod
    @static_multishape
    def get_key_from_var(var):

        from six import iteritems

        for (key, value) in iteritems(RandomVariable.__declared_vars):
            if value==var:
                return key
        return None


    def copy(self, swap_dict=None, observed=False):
        """ Build a new random variable with the same values.
        The underlying tensors or edward objects are copied as well.

        Args:
             swap_dict: random variables, variables, tensors, or operations to swap with.
             observed: determines if the new variable is observed or not.

        """

        new_var = getattr(sys.modules[self.__class__.__module__], type(self).__name__)()
        new_var.dist = ed.copy(self.dist, swap_dict)
        new_var.copied_from = self
        new_var.observed = False
        return new_var






# List of Python operators that we allow to override.
BINARY_OPERATORS = {
    # Binary.
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__div__",
    "__rdiv__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__mod__",
    "__rmod__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__xor__",
    "__rxor__",
#    "__getitem__",
    "__pow__",
    "__rpow__",
    "__matmul__",
    "__rmatmul__"

}
UNARY_OPERATORS = {
    # Unary.
#    "__invert__",
    "__neg__",
    "__abs__"
}




def __add_operator(cls, name, unary=False):

    import inferpy.models.deterministic

    if unary==False:
        def operator(self, other):

            res = inferpy.models.Deterministic()

            if isinstance(other, RandomVariable):
                res.base_object = getattr(self.base_object, name)(other.base_object)
            else:
                res.base_object = getattr(self.base_object, name)(other)
            return res
    else:
        def operator(self):
            res = inferpy.models.Deterministic()
            res.base_object = getattr(self.base_object, name)()
            return res


    operator.__doc__ = "documentation for "+name
    operator.__name__ = name
    setattr(cls, operator.__name__, operator)


for x in BINARY_OPERATORS:
    __add_operator(RandomVariable,x)


for x in UNARY_OPERATORS:
    __add_operator(RandomVariable,x, unary=True)



def __add_equal_operator():

    import inferpy.models.deterministic

    name = "equal"
    cls = RandomVariable
    def operator(self, other):

        res = inferpy.models.Deterministic()

        op1 = self.base_object
        op2 = other.base_object if isinstance(other, RandomVariable) else other

        res.base_object = tf.equal(op1,op2)

        return res


    operator.__doc__ = "documentation for " + name
    operator.__name__ = name
    setattr(cls, operator.__name__, operator)


__add_equal_operator()





def __add_getitem_operator_old():

    import inferpy.models.deterministic

    name = "__getitem__"
    cls = RandomVariable
    def operator(self, index):

        res = inferpy.models.Deterministic()


        if isinstance(index, inf.models.RandomVariable):
            if index.batches == 1:

                if self.batches > 1:
                    res_dist = tf.reshape(tf.gather(self.base_object, index.base_object, axis=1),
                                          (self.batches,1))
                else:
                    res_dist = tf.reshape(tf.gather(self.base_object, index.base_object, axis=0), (1, 1))

            else:

                res_dist = tf.reshape(tf.stack([self[index[n,0]][n,0].base_object
                                                for n in range(0, index.batches)]), (index.batches, 1))


        else:

            if np.ndim(index) < 1:
                index = [index]

            if len(np.shape(index)) < 2:
                In = range(0, self.batches) if len(index) < 2 else [index[0]]
                Id = [index[-1]]

                if self.batches>1:
                    res_dist = tf.gather(tf.gather(self.base_object, In, axis=0), Id, axis=1)
                else:
                    res_dist = tf.reshape(tf.gather(self.base_object, Id, axis=0), (1,1))
            else:

                res_dist = tf.reshape(tf.stack([self[0,index[n][0]].base_object
                                                for n in range(0, np.shape(index)[0])]), (self.batches,1))


        res.base_object = res_dist

        return res



    operator.__doc__ = "documentation for " + name
    operator.__name__ = name
    setattr(cls, operator.__name__, operator)





def __add_getitem_operator():

    import inferpy.models.deterministic

    name = "__getitem__"
    cls = RandomVariable
    def operator(self, index):

        res = inferpy.models.Deterministic()

        res_tf = self.dist

        if not isinstance(index, collections.Iterable):
            index = [index]

        axis = 0
        ndim = len(self.shape)

        for i in index:
            if isinstance(i, inf.models.RandomVariable):
                res_tf = tf.gather(res_tf, i.base_object, axis=axis)
            elif isinstance(i, tf.Tensor):
                res_tf = tf.gather(res_tf, i, axis=axis)
            else:

                I = tuple([slice(None,None,None) for x in range(0,axis)] + [i])
                res_tf = res_tf[I]

            ndim_new = len(res_tf._shape_as_list())

            if ndim != ndim_new:
                ndim = ndim_new
            else:
                axis = axis+1




        res.base_object = tf.reshape(res_tf, inf.fix_shape(res_tf.shape))

        return res





    operator.__doc__ = "documentation for " + name
    operator.__name__ = name
    setattr(cls, operator.__name__, operator)



__add_getitem_operator()




"""
        if np.ndim(index)<1:
            index = [index]
        elif np.ndim(index)>1:
            raise ValueError("wrong index dimensions "+str(index))


        res_dist = self.base_object

        for i in index:
            res_dist = tf.gather(res_dist, i if not isinstance(i, inf.models.RandomVariable) else i.dist)

        res.base_object = res_dist

        return res

"""