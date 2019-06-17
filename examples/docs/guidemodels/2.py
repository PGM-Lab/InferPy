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

>>> x.loc
<tf.Tensor 'randvar_0/Identity:0' shape=() dtype=float32>
"""


"""
>>> x.sample(tf_run=False)
<tf.Tensor 'randvar_0/sample/Reshape:0' shape=() dtype=float32>
"""


### 41





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
    x = inf.models.Normal(loc=tf.zeros(5), scale=1.)       # x.shape = [10,5]


y = x[7,4]                                              # y.shape = []

y2 = x[7]                                               # y2.shape = [5]

y3 = x[7,:]                                             # y2.shape = [5]

y4 = x[:,4]                                             # y4.shape = [10]



assert(y.shape==[])
assert(y2.shape==[5])
assert(y3.shape==[5])
assert(y4.shape==[10])


i = inf.Categorical(logits= tf.zeros(3))        # shape = []
mu = inf.Normal([5,1,-2], 0.)                   # shape = [3]
x = inf.Normal(mu[i], scale=1.)                 # shape = []



### 106



@inf.probmodel
def simple(mu=0):
    # global variables
    theta = inf.Normal(mu, 0.1, name="theta")

    # local variables
    with inf.datamodel():
        x = inf.Normal(theta, 1, name="x")







m = simple()

"""
>>> m.sample()
{'theta': -0.074800275, 'x': array([0.07758344], dtype=float32)}
"""



"""
>>> m.vars["theta"]
<inf.RandomVariable (Normal distribution) named theta/, shape=(), dtype=float32>
"""





m2 = simple(mu=5)

"""
>>> sess = tf.session()
>>> sess.run(m2.vars["x"].loc)
4.849595
"""




