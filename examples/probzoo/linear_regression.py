import numpy as np
import inferpy as inf
N=1000
x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])
##########


# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

# definition of a generic model
@inf.probmodel
def linear_reg(d):
    w0 = inf.Normal(0,1, name="w0")
    w = inf.Normal(0, 1, batch_shape=[d])

    with inf.datamodel():
        z = inf.Normal(tf.ones([k]),1, name="z")       # shape = [N,k]
        x = inf.Normal(z @ beta , 1, name="x")         # shape = [N,d]


# create an instance of the model

d = 5

