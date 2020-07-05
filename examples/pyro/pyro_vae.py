import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import inferpy as inf
import pyro
import torch
import tensorflow_probability.python.edward2 as ed

# number of components
k = 2
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 28 * 28
# number of observations (dataset size)
N = 1000
# batch size
M = 100
# digits considered
DIG = [0, 1, 2]
# minimum scale
scale_epsilon = 0.01
# inference parameters
num_epochs = 1000
learning_rate = 0.01

tf.reset_default_graph()
tf.set_random_seed(1234)
#29
from inferpy.data import mnist

# load the data
(x_train, y_train), _ = mnist.load_data(num_instances=N, digits=DIG)

mnist.plot_digits(x_train, grid=[5,5])


#38


### Model definition

class Decoder(torch.nn.Module):
    def __init__(self, k, d0, dx):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = torch.nn.Linear(k, d0)
        self.fc21 = torch.nn.Linear(d0, dx)
        # setup the non-linearities
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.relu(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        #loc_img = self.sigmoid(self.fc21(hidden))
        loc_img = self.fc21(hidden)
        return loc_img


class Encoder(torch.nn.Module):
    def __init__(self, k, d0, dx):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = torch.nn.Linear(dx, d0)
        self.fc21 = torch.nn.Linear(d0, k)
        self.fc22 = torch.nn.Linear(d0, k)
        # setup the non-linearities
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x k
        z_loc = self.fc21(hidden)
        z_scale = self.softplus(self.fc22(hidden))
        return z_loc, z_scale + scale_epsilon


class VAE(torch.nn.Module):
    def __init__(self, k=2, d0=100, dx=784):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(k, d0, dx)
        self.decoder = Decoder(k, d0, dx)
        self.k = k

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.k)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.k)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", pyro.distributions.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", pyro.distributions.Normal(loc_img, 1).to_event(1), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", pyro.distributions.Normal(z_loc, z_scale).to_event(1))





# 123

########### Inference

## setting up batched data
vae = VAE(k, d0, dx)

# Load data and set batch_size
train_loader = torch.utils.data.DataLoader(torch.tensor(x_train), batch_size=M, shuffle=False)

# setup the optimizer
adam_args = {"lr": learning_rate}
optimizer = pyro.optim.Adam(adam_args)

# setup the inference algorithm
svi = pyro.infer.SVI(vae.model, vae.guide, optimizer, loss=pyro.infer.Trace_ELBO())








train_elbo = []
pyro.clear_param_store()

# training loop
for epoch in range(num_epochs):
    epoch_loss = 0.
    for x in train_loader:
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train

    train_elbo.append(-total_epoch_loss_train)


    if (epoch % 10) == 0:
        print(total_epoch_loss_train)

# extract the posterior of z
postz = np.concatenate([
    vae.encoder.forward(x)[0].detach().numpy()
    for x in train_loader])


###
# for each input instance, plot the hidden encoding coloured by the number that it represents
markers = ["x", "+", "o"]
colors = [plt.get_cmap("gist_rainbow")(0.05),
          plt.get_cmap("gnuplot2")(0.08),
          plt.get_cmap("gist_rainbow")(0.33)]
transp = [0.9, 0.9, 0.5]

fig = plt.figure()

for c in range(0, len(DIG)):
    col = colors[c]
    plt.scatter(postz[y_train == DIG[c], 0], postz[y_train == DIG[c], 1], color=col,
                label=DIG[c], marker=markers[c], alpha=transp[c], s=60)
    plt.legend()

plt.show()


#####

## generate new digits
x_gen = vae.decoder.forward(torch.Tensor(postz[:M,:]))
mnist.plot_digits(x_gen.detach().numpy(), grid=[5,5])




# 200