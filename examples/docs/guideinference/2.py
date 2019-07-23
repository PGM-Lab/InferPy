import inferpy as inf
import tensorflow as tf

@inf.probmodel
def linear_reg(d):
    w0 = inf.Normal(0, 1, name="w0")
    w = inf.Normal(tf.zeros([d, 1]), 1, name="w")
    with inf.datamodel():
        x = inf.Normal(tf.ones(d), 2, name="x")
        y = inf.Normal(w0 + x @ w, 1.0, name="y")

m = linear_reg(2)

# Generate 100 samples for x and y random variables, with random variables w and w0 observed
data = m.prior(["x", "y"], data={"w0": 0, "w": [[2], [1]]}).sample(100)

# Define the qmodel and train
@inf.probmodel
def qmodel(d):
    qw0_loc = inf.Parameter(0., name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")
    qw_loc = inf.Parameter(tf.zeros([d, 1]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d, 1]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")

x_train = data["x"]
y_train = data["y"]

# set and run the inference
VI = inf.inference.VI(qmodel(2), epochs=10000)
m.fit({"x": x_train, "y": y_train}, VI)

# Now we can obtain the parameters of the hidden variables (after training)
m.posterior(["w", "w0"]).parameters()

# We can also generate new samples for the posterior distribution of the random variable x
post_data = m.posterior_predictive(["x", "y"]).sample()

# and we can check the log prob of the hidden variables, given the posterior sampled data
m.posterior(data=post_data).log_prob()