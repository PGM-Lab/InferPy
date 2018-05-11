import edward as ed
import inferpy as inf
from inferpy.models import Normal

d, N =  10, 200

# model definition
with inf.ProbModel() as m:

    #define the weights
    w0 = Normal(0,1)
    with inf.replicate(size=d):
        w = Normal(0, 1)

    # define the generative model
    with inf.replicate(size=N):
        x = Normal(0, 1, observed=True, dim=d)
        y = Normal(w0 + inf.matmul(x,w), 1.0, observed=True)

# toy data generation
x_train = Normal(loc=0, scale=1, dim=d).sample(N)
y_train = Normal(loc=5, scale=1, dim=1).sample(N)
data = {x.name: x_train, y.name: y_train}


# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))


