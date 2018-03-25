import edward as ed
import inferpy as inf
from inferpy.models import Normal

K, d, N = 5, 10, 200

# model definition
with inf.ProbModel() as m:
    #define the weights
    with inf.replicate(size=K):
        w = Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        z = Normal(0, 1, dim=K)
        x = Normal(inf.matmul(z,w),
                   1.0, observed=True, dim=d)

# data generation
x_train = Normal(loc=0, scale=1., dim=d).sample(N)
data = {x.name: x_train}

# compile and fit the model with training data
m.compile()
m.fit(data)

#extract the hidden representation from a set of observations
hidden_encoding = m.posterior(z)




from keras.models import Sequential
from keras.layers import Dense, Activation

M, dim_z, dim_x = 1000, 10, 100

#Define the decoder network
input_z  = keras.layers.Input(input_dim = dim_z)
layer = keras.layers.Dense(256, activation = 'relu')(input_z)
output_x = keras.layers.Dense(dim_x)(layer)
decoder_nn = keras.models.Model(inputs = input, outputs = output_x)

#define the generative model
with inf.replicate(size = N):
 z = Normal(0,1, dim = dim_z)
 x = Normal(loc = decoder_nn(z.value()), 1.0, observed = true)

#define the encoder network
input_x  = keras.layers.Input(input_dim = d_x)
layer = keras.layers.Dense(256, activation = 'relu')(input_x)
output_loc = keras.layers.Dense(dim_z)(layer)
output_scale = keras.layers.Dense(dim_z, activation = 'softplus')(layer)
encoder_loc = keras.models.Model(inputs = input, outputs = output_mu)
encoder_scale = keras.models.Model(inputs = input, outputs = output_scale)

#define the Q distribution
q_z = Normal(loc = encoder_loc(x.value()), scale = encoder_scale(x.value()))

#compile and fit the model with training data
probmodel.compile(infMethod = 'KLqp', Q = {z : q_z})
probmodel.fit(data)

#extract the hidden representation from a set of observations
hidden_encoding = probmodel.predict(x_pred, targetvar = z)