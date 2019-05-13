import inferpy as inf
import tensorflow as tf
import numpy as np


x = inf.Normal(loc = 0, scale = 1)

x.var
x.distribution

"""
>>> x = inf.Normal(loc = 0, scale = 1)

>>> x.var
<ed.RandomVariable 'randvar_0/' shape=() dtype=float32>

>>> x.distribution
<tfp.distributions.Normal 'randvar_0/' batch_shape=() event_shape=() dtype=float32>


"""


"""
>>> x.value
<tf.Tensor 'randvar_0/sample/Reshape:0' shape=() dtype=float32>

>>> x.sample()
-0.05060442
"""


### 33


# batch shape

x = inf.Normal(0,1, batch_shape=[3,2])                    # x.shape = [3,2]

x = inf.Normal(loc = [[0.,0.],[0.,0.],[0.,0.]], scale=1)  # x.shape = [3,2]

x = inf.Normal(loc = np.zeros([3,2]), scale=1)            # x.shape = [3,2]

x = inf.Normal(loc = 0, scale=tf.ones([3,2]))             # x.shape = [3,2]


# sample shape


x = inf.Normal(tf.ones([3,2]), 0, sample_shape=100)     # x.sample = [100,3,2]

with inf.datamodel(100):
    x = inf.Normal(tf.ones([3, 2]), 0)                  # x.sample = [100,3,2]



# event shape

x = inf.MultivariateNormalDiag(loc=[1., -1], scale_diag=[1, 2.])



### 63

"""

>>> x.event_shape
TensorShape([Dimension(2)])

>>> x.batch_shape
TensorShape([])

>>> x.sample_shape
TensorShape([])



"""




###  83

with inf.datamodel(size=10):
    x = inf.models.Normal(loc=0., scale=1., batch_shape=[5])       # x.shape = [10,5]


y = x[7,4]                                              # y.shape = []

y2 = x[7]                                               # y2.shape = [5]

y3 = x[7,:]                                             # y2.shape = [5]

y4 = x[:,4]                                             # y4.shape = [10]



assert(y.shape==[])
assert(y2.shape==[5])
assert(y3.shape==[5])
assert(y4.shape==[10])



### 106







