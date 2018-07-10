import edward as ed
import inferpy as inf
from inferpy.models import Normal, Bernoulli, Categorical

d, N =  10, 500

# model definition
with inf.ProbModel() as m:

    #define the weights
    w0 = Normal(0,1)
    w = Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = Normal(0, 1, observed=True, dim=d)
        y = Bernoulli(logits=w0+inf.dot(x, w), observed=True)


# toy data generation
x_train = Normal(loc=0, scale=1, dim=d).sample(N)
y_train = Bernoulli(probs=0.4).sample(N)
data = {x.name: x_train, y.name: y_train}

# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))




