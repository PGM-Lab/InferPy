import inferpy.util


class RandomVariable(object):
    """Base class for random variables.
    """

    def __init__(self, dist):
        self._dist = dist

    @property
    def dist(self):
        """Underlying Edward object"""
        return self._dist

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


    def sample(self, v):
        """ Method for obaining a sample of shape v"""
        return inferpy.util.runtime.tf_sess.run(self.dist.sample(v))


