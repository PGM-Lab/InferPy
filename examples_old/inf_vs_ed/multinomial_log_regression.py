import edward as ed
import inferpy as inf
import numpy as np
import tensorflow as tf

d, N =  10, 500

#number of classes
K = 3

# toy data generation
x_train = inf.models.Normal(loc=0, scale=1, dim=d).sample(N)
y_train = inf.models.Bernoulli(probs=np.random.rand(K)).sample(N)


###### Edward ########

# define the weights
w0 = ed.models.Normal(loc=tf.zeros(K), scale=tf.ones(K))
w = ed.models.Normal(loc=tf.zeros([K,d]), scale=tf.ones([K,d]))

# define the generative model
x = ed.models.Normal(loc=tf.zeros([N,d]), scale=tf.ones([N,d]))
y = ed.models.Normal(loc=w0 + tf.matmul(x, w, transpose_b=True), scale=tf.ones([N,K]))

# compile and fit the model with training data
qw = ed.models.Normal(loc=tf.Variable(tf.random_normal([K,d])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([K,d]))))
qw0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([K])),
                       scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

inference = ed.KLqp({w: qw, w0: qw0}, data={x: x_train, y: y_train})
inference.run()

# print the posterior distributions
print([qw.loc.eval(), qw0.loc.eval()])





###### Inferpy ########

# model definition
with inf.ProbModel() as m:

    #define the weights
    w0 = inf.models.Normal(0,1, dim=K)

    with inf.replicate(size=K):
        w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Bernoulli(logits = w0 + inf.matmul(x, w, transpose_b=True), observed=True)



data = {x.name: x_train, y.name: y_train}


# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))




