# import the package for using it

import inferpy as inf
import edward as ed
import numpy as np
from six import iteritems


sess = ed.get_session()


N = 1000     # number of observations

# toy data generation

x_train = inf.models.Normal(loc=10, scale=3, dim=1).sample(N)
y_train = x_train*2 + inf.models.Normal(loc=1, scale=0.1, dim=1).sample(N)


# model definition
with inf.ProbModel() as m:
    # prior (latent variable)
    beta = inf.models.Normal(loc=0, scale=1, name="beta")
    w = inf.models.Normal(loc=0, scale=1, name = "w")
    b = inf.models.Normal(loc=0, scale=1, name="b")
    betaz = inf.models.Normal(loc=0, scale=1, name="beta")



    # observed variable
    with inf.replicate(size=N):

        z = inf.models.Normal(loc=betaz, scale=1, observed=False, name="z")
        x = inf.models.Normal(loc=beta+z, scale=1, observed=True, name="x")
        y = inf.models.Normal(loc = w*x+b+z, scale=1, observed=True, name="y")


data = {x.name : x_train, y.name : y_train}

m.compile()
m.fit(data)


x_test = inf.models.Normal(loc=10, scale=3, dim=1).sample(N)
y_test = x_test*2 + inf.models.Normal(loc=1, scale=0.1, dim=1).sample(N)


# predict
y_pred = m.predict(y, data={x : x_test}).loc


# evaluate the predicted data y=y_pred given that x=x_test
mse = inf.evaluate('mean_squared_error', data={x: x_test, y: y_pred}, output_key=y)

