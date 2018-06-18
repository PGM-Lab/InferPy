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
from inferpy.util import static_multishape
import inferpy as inf
import tensorflow as tf
import edward as ed
import numpy as np





class RandomVariable(object):
    """Base class for random variables.
    """


    __declared_vars = {}

    def __init__(self, base_object=None, observed=False):

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
        return self.base_object.shape.as_list()[-1]

    @property
    def batches(self):
        """ Number of batches of the variable"""

        dist_shape = self.base_object.shape.as_list()
        if len(dist_shape)>1:
            return dist_shape[-2]
        return 1

    @property
    def shape(self):
        """ shape of the variable, i.e. (batches, dim)"""
        return self.base_object.shape.as_list()

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
            s = tf.reshape(s, shape=(size,self.dim))

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


    def is_generic_variable(self):
        return isinstance(self._base_object, ed.RandomVariable) == False


    def get_replicate_list(self):
        return [r  for r in inf.replicate.get_all_replicate() if self in r.varlist]


    def get_local_hidden(self):

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

        new_var = getattr(inf.models, type(self).__name__)()
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

