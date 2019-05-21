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


# number of components
k = 1
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 2
# number of observations (dataset size)
N = 1000


@inf.probmodel
def vae(k, d0, dx, decoder):

    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5, 1., name="z")    # shape = [N,k]
        output = decoder(z,d0,dx)
        x_loc = output[:,:dx]
        x_scale = tf.nn.softmax(output[:,dx:])
        x = inf.Normal(x_loc, x_scale, name="x")   # shape = [N,d]


def decoder(z,d0,dx):   # k -> d0 -> 2*dx
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * dx)


# Q-model  approximating P
def encoder(x, d0, k): # dx -> d0 -> 2*k
    h0 = tf.layers.dense(tf.constant(x), d0, tf.nn.relu)
    return tf.layers.dense(h0, 2*k)

@inf.probmodel
def qmodel(k, d0, x, encoder):

    output = encoder(x, d0, k)
    qz_loc = output[:, :k]
    qz_scale = tf.nn.softmax(output[:, k:])

    qz = inf.Normal(qz_loc, qz_scale, name="z")


# create an instance of the model
m = vae(k,d0,dx, decoder)
q = qmodel(k,d0,x_train,encoder)

# set the inference algorithm
VI = inf.inference.VI(q, epochs=5000)

# learn the parameters
m.fit({"x": x_train}, VI)


#extract the hidden representation
hidden_encoding = m.posterior["z"]
sess = inf.get_session()
print(sess.run(hidden_encoding.loc))
