### Setting up

import torch
import pyro
from pyro.distributions import Normal, Binomial
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal

d = 2
N = 1000





### 17


### Model definition ###

def log_reg(x_data=None, y_data=None):
    w = pyro.sample("w", Normal(torch.zeros(d), torch.ones(d)))
    w0 = pyro.sample("w0", Normal(0., 1.))

    with pyro.plate("map", N):
        x = pyro.sample("x", Normal(torch.zeros(d), 2).to_event(1), obs=x_data)
        logits = (w0 + x @ torch.FloatTensor(w)).squeeze(-1)
        y = pyro.sample("pred", Binomial(logits = logits), obs=y_data)

    return x,y


qmodel = AutoDiagonalNormal(log_reg)


### 37

#### Sample from prior model

sampler = pyro.condition(log_reg, data={"w0": 0, "w": [2,1]})
x_train, y_train = sampler()

#### Inference

optim = Adam({"lr": 0.1})
svi = SVI(log_reg, qmodel, optim, loss=Trace_ELBO(), num_samples=10)


num_iterations = 10000
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_train, y_train)
    print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train)))


#### Usage of the inferred model


# Print the parameters
w_post = qmodel()["w"]
w0_post = qmodel()["w0"]

print(w_post, w0_post)

# Sample from the posterior
sampler_post = pyro.condition(log_reg, data={"w0": w0_post, "w": w_post})
x_gen, y_gen = sampler_post()

print(x_gen, y_gen)




##### Plot the results

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
for c in [0, 1]:
    x_gen_c = x_gen[y_gen.flatten() == c, :]
    plt.plot(x_gen_c[:, 0], x_gen_c[:, 1], 'bx' if c == 0 else 'rx')

plt.show()



### 90