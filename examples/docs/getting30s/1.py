import inferpy as inf
import tensorflow as tf

# number of components
k = 1
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 2
# number of observations (dataset size)
N = 1000


@inf.probmodel
def nlpca(k, d0, dx, decoder):

    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5, 1., name="z")    # shape = [N,k]
        output = decoder(z,d0,dx)
        x_loc = output[:,:dx]
        x_scale = tf.nn.softmax(output[:,dx:])
        x = inf.Normal(x_loc, x_scale, name="x")   # shape = [N,d]


###########  25

def decoder(z,d0,dx):
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * dx)


######### 32

# Q-model  approximating P

@inf.probmodel
def qmodel(k):
    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k])*0.5, name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]),name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")



####### 46

# create an instance of the model
m = nlpca(k,d0,dx, decoder)

# Sample from priors
samples = m.sample()


#### NOT showing  55
import numpy as np
x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])


###### 60

# set the inference algorithm
VI = inf.inference.VI(qmodel(k), epochs=5000)

# learn the parameters
m.fit({"x": x_train}, VI)


#### 69

#extract the hidden representation
hidden_encoding = m.posterior["z"]
print(hidden_encoding.sample())


####