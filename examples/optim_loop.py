import tensorflow as tf
from tensorflow_probability import edward2 as ed
import inferpy as inf



@inf.probmodel
def simple():
    theta = inf.Normal(0, 1, name="theta")
    with inf.datamodel():
        x = inf.Normal(theta, 2, name="x")


@inf.probmodel
def q_model():
    qtheta_loc = inf.Parameter(1., name="qtheta_loc")
    qtheta_scale = tf.math.softplus(inf.Parameter(1., name="qtheta_scale"))

    qtheta = inf.Normal(qtheta_loc, qtheta_scale, name="theta")



## example of use ###
# generate training data
N = 1000
sess = tf.Session()
x_train = sess.run(ed.Normal(5., 2.).distribution.sample(N))


import inferpy.inference.loss_functions.elbo


m = simple()

q = q_model()



inf.util.runtime.set_tf_run(False)

loss_tensor = inf.inference.loss_functions.elbo.ELBO(m,q, {"x": x_train})
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss_tensor)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    t = []
    for i in range(0,100):
        sess.run(train)
        t += [sess.run(loss_tensor)]
        print(t[-1])

    posterior_qvars = {name: qv.build_in_session(sess) for name, qv in q._last_expanded_vars.items()}



