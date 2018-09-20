import edward as ed
import inferpy as inf
import tensorflow as tf

d, N =  10, 500

x_train = inf.models.Normal(loc=0, scale=1, dim=d).sample(N)
y_train = inf.models.Bernoulli(probs=0.4).sample(N)


########### Edward ############


# define the weights
w0 = ed.models.Normal(loc=tf.zeros(1), scale=tf.ones(1))
w = ed.models.Normal(loc=tf.zeros(d), scale=tf.ones(d))

# define the generative model
x = ed.models.Normal(loc=tf.zeros([N,d]), scale=tf.ones([N,d]))
y = ed.models.Bernoulli(logits=ed.dot(x, w) + w0)

# compile and fit the model with training data
qw = ed.models.Normal(loc=tf.Variable(tf.random_normal([d])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([d]))))
qw0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([1])),
                       scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, w0: qw0}, data={x: x_train, y: y_train.reshape(N)})
inference.run()

# print the posterior distributions
print([qw.loc.eval(), qw0.loc.eval()])




########### Inferpy ###########

# model definition
with inf.ProbModel() as m:

    #define the weights
    w0 = inf.models.Normal(0,1)
    w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Bernoulli(logits=w0+inf.dot(x, w), observed=True)


# toy data generation
data = {x: x_train, y: y_train}

# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))




