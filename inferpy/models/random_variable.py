import inferpy.util
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
        return self.__observed

    @observed.setter
    def observed(self,observed):
        self.__observed=observed


    @dist.setter
    def dist(self, dist):
        """ Set the Underlying Edward object"""

        if isinstance(dist, ed.models.RandomVariable)==False:
            raise ValueError("Type of input distribution is nor correct")

        self._dist = dist


    @tf_run_wrapper
    def sample(self, size=1):
        """ Method for obaining a samples"""
        s = self.dist.sample(size)


        if self.batches == 1:
            s = tf.reshape(s, [size, self.dim])

        return s


    @tf_run_wrapper
    def prob(self, v):
        return self.dist.prob(tf.cast(v, tf.float64))

    @tf_run_wrapper
    def log_prob(self, v):
        return self.dist.log_prob(tf.cast(v, tf.float64))

    @tf_run_wrapper
    def prod_prob(self, v):
        return tf.reduce_prod(self.dist.prob(tf.cast(v, tf.float64)))

    @tf_run_wrapper
    def sum_log_prob(self, v):
        return tf.reduce_sum(self.dist.log_prob(tf.cast(v, tf.float64)))

    def __repr__(self):
        return "<inferpy RandomVariable '%s' shape=%s dtype=%s>" % (
            self.name, self.shape, self.dist.dtype.name)



