### Setting up

from tensorflow_probability import edward2 as ed
import tensorflow as tf

d = 2
N = 50000




#12

### Model definition ####

def log_reg(d, N, w_init=(1, 1), x_init=(0, 1)):
    w = ed.Normal(loc=tf.ones([d], dtype="float32") * w_init[0], scale=1. * w_init[1], name="w")
    w0 = ed.Normal(loc=1. * w_init[0], scale=1. * w_init[1], name="w0")

    x = ed.Normal(loc=tf.ones([N, d], dtype="float32") * x_init[0], scale=1. * x_init[1], name="x")
    y = ed.Bernoulli(logits=tf.tensordot(x, w, axes=[[1], [0]]) + w0, name="y")

    return x, y, (w, w0)


def qmodel(d, N):
    qw_loc = tf.Variable(tf.ones([d]))
    qw_scale = tf.math.softplus(tf.Variable(tf.ones([d])))
    qw0_loc = tf.Variable(1.)
    qw0_scale = tf.math.softplus(tf.Variable(1.))

    qw = ed.Normal(loc=qw_loc, scale=qw_scale, name="qw")
    qw0 = ed.Normal(loc=qw0_loc, scale=qw0_scale, name="qw0")
    return qw, qw0




#39

##### Sample from prior model

# Training data is generated from the same model, but with the priors variables
# set to given values w_sampling and w0_sampling. This is done with ed.interception


def set_values(**model_kwargs):
    """Creates a value-setting interceptor."""

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        else:
            print(f"set_values not interested in {name}.")
        return ed.interceptable(f)(*args, **kwargs)

    return interceptor


with ed.interception(set_values(w=[2, 1], w0=0)):
    generate = log_reg(d, N, x_init=(2, 10))

with tf.Session() as sess:
    x_train, y_train, _ = sess.run(generate)



#### Inference

qw, qw0 = qmodel(d, N)

with ed.interception(set_values(w=qw, w0=qw0, x=x_train, y=y_train)):
    post_x, post_y, (post_w, post_w0) = log_reg(d, N)

energy = tf.reduce_sum(post_x.distribution.log_prob(post_x.value)) +  \
                    tf.reduce_sum(post_y.distribution.log_prob(y_train)) + \
                    tf.reduce_sum(post_w.distribution.log_prob(qw.value)) +  \
                    tf.reduce_sum(post_w0.distribution.log_prob(qw0.value))

entropy = -(tf.reduce_sum(qw.distribution.log_prob(qw.value)) + \
            tf.reduce_sum(qw0.distribution.log_prob(qw0.value)))


# ELBO definition
elbo = energy + entropy


# Optimization loop
optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(-elbo)

init = tf.global_variables_initializer()

t = []
num_epochs = 10000
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_epochs):
        sess.run(train)
        if i % 5 == 0:
            t.append(sess.run([elbo]))

            if i % 50 == 0:
                print(sess.run(elbo))

    w_post = sess.run(qw.distribution.loc)
    w0_post = sess.run(qw0.distribution.loc)


#### Usage of the inferred model

# Print the parameters
print(w_post, w0_post)

# Sample form the posterior
with ed.interception(set_values(w=w_post, w0=w0_post)):
    generate = log_reg(d, N, x_init=(0, 0))

with tf.Session() as sess:
    x_gen, y_gen, _ = sess.run(generate)

print(x_gen, y_gen)





##### Plot the results

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
for c in [0, 1]:
    x_gen_c = x_gen[y_gen == c, :]
    plt.plot(x_gen_c[:, 0], x_gen_c[:, 1], 'bx' if c == 0 else 'rx')

plt.show()



# 146


