import inferpy as inf
import tensorflow as tf
import numpy as np


# different ways of declaring 1 batch of 5 Normal distributions

x = inf.models.Normal(loc = 0, scale=1, dim=5)         # x.shape = [5]

x = inf.models.Normal(loc = [0, 0, 0, 0, 0], scale=1)  # x.shape = [5]

x = inf.models.Normal(loc = np.zeros(5), scale=1)      # x.shape = [5]

x = inf.models.Normal(loc = 0, scale=tf.ones(5))       # x.shape = [5]




with inf.replicate(size=10):
    x = inf.models.Normal(loc=0, scale=1, dim=10)       # x.shape = [10,5]