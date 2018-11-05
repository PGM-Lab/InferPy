import inferpy as inf
import tensorflow as tf
import numpy as np


# different ways of declaring 1 batch of 5 Normal distributions

x = inf.models.Normal(loc = 0, scale=1, dim=5)         # x.shape = [5]

x = inf.models.Normal(loc = [0, 0, 0, 0, 0], scale=1)  # x.shape = [5]

x = inf.models.Normal(loc = np.zeros(5), scale=1)      # x.shape = [5]

x = inf.models.Normal(loc = 0, scale=tf.ones(5))       # x.shape = [5]




with inf.replicate(size=10):
    x = inf.models.Normal(loc=0, scale=1, dim=5)       # x.shape = [10,5]

x = inf.models.Normal(loc=0, scale=1, dim=[10,5])       # x.shape = [10,5]

y = x[7,4]                                              # y.shape = [1]

y2 = x[7]                                               # y2.shape = [5]

y3 = x[7,:]                                             # y2.shape = [5]

y4 = x[:,4]                                             # y4.shape = [10]



z = inf.models.Categorical(logits = np.zeros(5))
yz = inf.models.Normal(loc=x[0,z], scale=1)            # yz.shape = [1]


### 38







