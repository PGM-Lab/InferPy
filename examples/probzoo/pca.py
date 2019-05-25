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
def pca(k,d):
    beta = inf.Normal(loc=np.zeros([k,d]),
                   scale=1, name="beta")               # shape = [k,d]

    with inf.datamodel():
        z = inf.Normal(tf.ones([k]),1, name="z")       # shape = [N,k]
        x = inf.Normal(z @ beta , 1, name="x")         # shape = [N,d]


# create an instance of the model
m = pca(k=1,d=2)



@inf.probmodel
def qmodel(k,d):
    qbeta_loc = inf.Parameter(tf.zeros([k,d]), name="qbeta_loc")
    qbeta_scale = tf.math.softplus(inf.Parameter(tf.ones([k,d]),
                                                 name="qbeta_scale"))

    qbeta = inf.Normal(qbeta_loc, qbeta_scale, name="beta")

    with inf.datamodel():
        qz_loc = inf.Parameter(np.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]),
                                                  name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")



VI = inf.inference.VI(qmodel(k=1,d=2), epochs=2000)
m.fit({"x": x_train}, VI)

sess = inf.get_session()
post_z = sess.run(m.posterior["z"].loc)
print(post_z)




