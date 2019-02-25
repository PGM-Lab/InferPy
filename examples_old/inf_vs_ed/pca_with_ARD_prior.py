import edward as ed
import inferpy as inf
import tensorflow as tf

K, d, N = 5, 10, 200

x_train = inf.models.Normal(loc=0, scale=1., dim=d).sample(N)



### Edward ####


# define the weights
w = ed.models.Normal(loc=tf.zeros([K,d]), scale=tf.ones([K,d]))

sigma = ed.models.InverseGamma(1.,1.)

# define the generative model
z = ed.models.Normal(loc=tf.zeros([N,K]), scale=tf.ones([N,K]))
x = ed.models.Normal(loc=tf.matmul(z,w), scale=sigma)

# compile and fit the model with training data
qw = ed.models.Normal(loc=tf.Variable(tf.random_normal([K,d])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([K,d]))))


inference = ed.KLqp({w: qw}, data={x: x_train})
inference.run()

# print the posterior distributions
print([qw.loc.eval()])



### InferPy ####

# model definition
with inf.ProbModel() as m:
    #define the weights
    with inf.replicate(size=K):
        w = inf.models.Normal(0, 1, dim=d)

    sigma = inf.models.InverseGamma(1.0,1.0)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Normal(0, 1, dim=K)
        x = inf.models.Normal(inf.matmul(z,w),
                   sigma, observed=True, dim=d)

# toy data generation
data = {x.name: x_train}


# compile and fit the model with training data
m.compile()
m.fit(data)

#extract the hidden representation from a set of observations
hidden_encoding = m.posterior(z)

