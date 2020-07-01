import numpy as np
import pandas as pd
import torch
import pyro
from pyro.distributions import Normal, Uniform, Delta, Gamma, Binomial
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.optim as optim
from pyro.contrib.autoguide import AutoDiagonalNormal

x_data = torch.rand(100,2)-0.5
w_true = torch.ones(2,1)
y_data = torch.tensor(1/(1 + torch.exp(-torch.mm(x_data,w_true)))>0.5).double().squeeze()

def model(x_data, y_data):
    # weight and bias priors
    with pyro.plate("plate_w", 2):
        w = pyro.sample("w", Normal(torch.zeros(1,1), torch.ones(1,1)))

    b = pyro.sample("b", Normal(0., 1000.))
    with pyro.plate("map", len(x_data)):
        # Compute logits (i.e. log p(x=0)/p(x=1)) as a linear combination between data and weights.
        logits = (b + torch.mm(x_data,torch.t(w))).squeeze(-1)
        # Define a Binomial distribution as the observed value parameterized by the logits.
        pyro.sample("pred", Binomial(logits = logits), obs=y_data)

optim = Adam({"lr": 0.1})
guide = AutoDiagonalNormal(model)
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=10)

num_iterations = 3000
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 500 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))


## Print parameters
guide()['b'].mean()
guide()['w'].mean()