import inferpy as inf
import numpy as np
import tensorflow as tf

# definition of a generic model
@inf.probmodel
def pca(k,d):
    w = inf.Normal(loc=np.zeros([k,d]), scale=1, name="w")      # shape = [k,d]
    with inf.datamodel():
        z = inf.Normal(np.ones([k]),1, name="z")                # shape = [N,k]
        x = inf.Normal(z @ w , 1, name="x")                     # shape = [N,d]


# create an instance of the model
m = pca(k=1,d=2)


#### data (do not show)
# generate training data
N = 1000
sess = tf.Session()
x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])



### encapsulated inference



@inf.probmodel
def qmodel(k,d):
    qw_loc = inf.Parameter(tf.ones([k,d]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")

    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(qz_loc, qz_scale, name="z")



# set the inference algorithm
VI = inf.inference.VI(qmodel(k=1,d=2), epochs=1000)

# run the inference
m.fit({"x": x_train}, VI)

print(m.posterior['w'].sample())

#
# ## ELBO definition
#
# from tensorflow_probability import edward2 as ed
#
# # define custom elbo function
# def custom_elbo(pmodel, qmodel, sample_dict):
#     # create combined model
#     plate_size = pmodel._get_plate_size(sample_dict)
#
#     # expand the qmodel (just in case the q model uses data from sample_dict, use interceptor too)
#     with ed.interception(inf.util.interceptor.set_values(**sample_dict)):
#         qvars, _ = qmodel.expand_model(plate_size)
#
#     # expand de pmodel, using the intercept.set_values function, to include the sample_dict and the expanded qvars
#     with ed.interception(inf.util.interceptor.set_values(**{**qvars, **sample_dict})):
#         pvars, _ = pmodel.expand_model(plate_size)
#
#     # compute energy
#     energy = tf.reduce_sum([tf.reduce_sum(p.log_prob(p.value)) for p in pvars.values()])
#
#     # compute entropy
#     entropy = - tf.reduce_sum([tf.reduce_sum(q.log_prob(q.value)) for q in qvars.values()])
#
#     # compute ELBO
#     ELBO = energy + entropy
#
#     # This function will be minimized. Return minus ELBO
#     return -ELBO
#
#
#
# # set the inference algorithm
# VI = inf.inference.VI(qmodel(k=1,d=2), loss=custom_elbo, epochs=1000)
#
# # run the inference
# m.fit({"x": x_train}, VI)
#
#
#
# ### optimization loop


