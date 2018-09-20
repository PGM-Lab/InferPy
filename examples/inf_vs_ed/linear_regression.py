import edward as ed
import inferpy as inf
import numpy as np
import tensorflow as tf

d, N =  5, 20000



# toy data generation
x_train = inf.models.Normal(loc=10, scale=5, dim=d).sample(N)
y_train = np.matmul(x_train, np.array([10,10,0.1,0.5,2]).reshape((d,1))) \
          + inf.models.Normal(loc=0, scale=5, dim=1).sample(N)



############################## InferPy #################################################

# model definition
with inf.ProbModel() as model:

    # define the weights
    w0 = inf.models.Normal(0,1)
    w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Normal(w0 + inf.dot(x,w), 1.0, observed=True)


# compile and fit the model with training data
model.compile()
data = {x: x_train, y: y_train}
model.fit(data)

# print the posterior distributions
print(m.posterior([w, w0]))



############################## Edward ##################################################

# define the weights
w0 = ed.models.Normal(loc=tf.zeros(1), scale=tf.ones(1))
w = ed.models.Normal(loc=tf.zeros(d), scale=tf.ones(d))

# define the generative model
x = ed.models.Normal(loc=tf.zeros([N,d]), scale=tf.ones([N,d]))
y = ed.models.Normal(loc=ed.dot(x, w) + w0, scale=tf.ones(N))

# compile and fit the model with training data
qw = ed.models.Normal(loc=tf.Variable(tf.random_normal([d])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([d]))))
qw0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([1])),
                       scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, w0: qw0}, data={x: x_train, y: y_train.reshape(N)})
inference.run()

# print the posterior distributions
print([qw.loc.eval(), qw0.loc.eval()])
