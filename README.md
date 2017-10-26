# INFERPY: A Python Library for Probabilistic Modelling

INFERPY is a high-level API for probabilistic modelling written in Python and capable of running on top of Edward, Tensorflow and Apache Spark. INFERPY's API is strongly inspired by Keras and it has a focus on enabling flexible data processing, simple probablistic modelling, scalable inference and robust model validation. 

Use INFERPY is you need a probabilistic programming language that:
 - Has a simple and user friendly API (inspired by Keras).
 - Allows for easy and fast prototyping of simple probabilistic models or complex probabilistics constructs containing deep neural networks (by relying on Edward).   
 - Run seamlessly on CPU and GPU (by relying on Tensorflow). 
 - Process seamlessly small data sets or large distributed data sets (by relying on Apache Spark). . 

----

## Getting Started: 30 seconds to INFERPY 

The core data structures of INFERPY is a a **probabilistic model**, defined as a set of **random variables** with a conditional independence structure. Like in Edward, a **random varible** is an object parameterized by a set of tensors. 

Let's look at a simple examle. We start defining hhe **prior** over the parameters of a **mixture of Gaussians** model: 


```python
import numpy as np
import inferpy as inf
from inferpy.models import Normal, InverseGamma, Dirichlet

# K defines the number of components. 
K=10
#Prior for the means of the Gaussians 
mu = Normal(loc = 0, scale = 1, shape=[K,d])
#Prior for the precision of the Gaussians 
sigma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
#Prior for the mixing proportions
p = Dirichlet(np.ones(K))
```
The **shape** argument in the constructor defines the number (and dimension) of variables contained in a random variable object. For example, **mu** contains K*d varaibles laid in a Kxd matrix. 

INFERPY supports the definition of **plateau notation** by using the construct ```with inf.replicate(size = N) ```, which replicates N times the random variables enclosed within this anotator. This is usefuel when defining the model for the data:

```python
# Number of observations
N = 1000
#data Model
with inf.replicate(size = N)
    # Sample the component of the mixture this sample belongs to. 
    # This is a latent indicator variable that can not be observed
    z_n = Multinomial(probs = p)
    # Sample the observed value from the Gaussian of the selected component.  
    x_n = Normal(loc = inf.gather(mu,z_n), scale = inf.gather(sigma,z_n), observed = true)
```
As commented above, the variable z_n and x_n are surrounded by a **with** statement to inidicate that the defined random variables will be reapeatedly used in each data sample.

Once the random variables of the  model are defined, the probablitic model itself can be created and compiled. The probabilistic model defines a joint probability distribuiton over all these random variables.  
```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [p,mu,sigma,z_n, x_n]) 
probmodel.compile(infMethod = 'KLqp'')
```
During the model compilation we specify different inference methods that will be used to learn the model. 

```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [p,mu,sigma,z_n, x_n]) 
probmodel.compile(infMethod = 'MCMC')
```

The inference method can be further configure. But, as in Keras, a core principle is to try make things reasonbly simple, while allowing the user the full control if needed. 

```python
from keras.optimizers import SGD
probmodel = ProbModel(vars = [p,mu,sigma,z_n, x_n]) 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
infklqp = inf.inference.KLqp(optimizer = sgd, dataSize = N)
probmodel.compile(infMethod = infklqp)
```

Every random variable object is equipped with methods such as *log_prob()* and *sample()*. Similarly, a probabilistic model is also equipped with the same methods. Then, we can sample data from the model anbd compute the log-likelihood of a data set:
```python
data = probmodel.sample(size = 1000)
log_like = probmodel.log_prob(data)
```

Of course, you can fit your model with a given data set:
```python
probmodel.fit(data_training, epochs=10)
```
Update your probablistic model with new data using the Bayes' rule:
```python
probmodel.update(new_data_training)
```
Query the posterior over a given random varible:
```python
x_post = probmodel.posterior(mu)
```
Evaluate your model according to a given metric:
```python
metrics = probmodel.evaluate(test_data, metrics = ['log_likelihood'])
```
Or compute predicitons on new data
```python
cluster_assignments = probmodel.predict(test_data, targetvar = h)
```

----

## Guiding Principles

- INFERPY's probability distribuionts are mainly inherited from TensorFlow Distribuitons package. INFERPY's API is fully compatible with tf.distributions' API. The 'shape' argument was added as a simplifing option when defining multidimensional distributions. 
- INFERPY directly relies on top of Edward's inference engine and includes all the inference algorithms included in this package. As Edward's inference engine relies on TensorFlow computing engine, INFERPY also relies on it too.  
- INFERPY seamsly process data contained in a numpy array, Tensorflow's tensor, Tensorflow's Dataset (tf.Data API) or Apache Spark's DataFrame. 
- INFERPY also includes novel distributed statistical inference algorithms by combining Tensorflow and Apache Spark computing engines. 

----

## Getting Started
### Guide to Building Probabilistic Models

INFERPY focuses on *hirearchical probabilistic models* which usually are structured in two different layers:

- A **prior model** defining a joint distribution $p(\theta)$ over the global parameters of the model, $\theta$.  
- A **data or observation model** defining a joint conditional distribution $p(x,h|\theta)$ over the observed quantities $x$ and the the local hidden variables $h$ governing the observation $x$. This data model should be specified in a single-sample basis. There are many models of interest without local hidden variables, in that case we simply specify the conditional $p(x|\theta)$. More flexible ways of defining the data model can be found in ?. 

This is how a mixture of Gaussians models is denfined in INFERPY: 
```python
import numpy as np
import inferpy as inf
from inferpy.models import Normal, InverseGamma, Dirichlet

# K defines the number of components. 
K=10
#Prior for the means of the Gaussians 
mu = Normal(loc = 0, scale = 1, shape=[K,d])
#Prior for the precision of the Gaussians 
sigma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
#Prior for the mixing proportions
theta = Dirichlet(np.ones(K))

# Number of observations
N = 1000
#data Model
with inf.replicate(size = N)
    # Sample the component of the mixture this sample belongs to. 
    # This is a latent indicator variable that can not be observed
    z_n = Multinomial(probs = theta)
    # Sample the observed value from the Gaussian of the selected component.  
    x_n = Normal(loc = inf.gather(mu,z_n), scale = inf.gather(sigma,z_n), observed = true)

#Probabilistic Model
probmodel = ProbModel(prior = [p,mu,sigma,z_n,x_n]) 
probmodel.compile()
```

The ```with inf.replicate(size = N)``` sintaxis is used to replicate the random variables contained within this construct. It follows from the standard *plateau notation* to define the data generation part of a probabilistic model. Internally, ```with inf.replicate(size = N)``` construct modifies the random variable shape by adding an extra dimension. For the above example, z_n's shape is [N,1], and x_n's shape is [N,d].  

Following Edward's approach, a random variable $x$  is an object parametrized by a tensor $\theta$ (i.e. a TensorFlow's tensor or numpy's ndarray). The number of random variables in one object is determined by the dimensions of its parameters (like in Edward) or by the 'shape' argument (inspired by PyMC3 and Keras):
```python
# vector of 5 univariate standard normals
x  = Normal(loc = np.zeros(5), scale = np.ones(5)) 

# vector of 5 univariate standard normals
x = Normal (loc = 0, scale = 1, shape = [5,1])
```
The ```with inf.replicate(size = N)``` sintaxis can  also be used to define multi-dimensional objects, the following code is also equivalent to the above ones:
```python
# vector of 5 univariate standard normals
with inf.replicate(size = 5)
    x = Normal (loc = 0, scale = 1)
```
More detailed inforamtion about the semantics of ```with inf.replicate(size = N)``` can be found in ?. Examples of using this construct to define more expressive and complex models can be found in ?. 


Multivariate distributions can be defined similarly. Similiarly to Edward's approach, the multivariate dimension is the innermost (right-most) dimension of the parameters. 
```python
# 2 x 3 matrix of K-dimensional multivariate normals
x  = MultivariateNormal(loc = np.zeros((2,3,K)), scale = np.ones((2,3,K,K)), observed = true) 

# 2 x 3 matrix of K-dimensional multivariate normals
y = MultivariateNormal (loc = np.zeros(K), scale = np.ones((K,K)), shape = [2,3], observed = true)
```

The argument **observed = true** in the constructor of a random variable is used to indicate whether a variable is observable or not.  

A **probabilistic model** defines a joint distribution over observable and non-observable variables.  $p(theta,mu,sigma,z_n, x_n)$ for the running example, 

```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
probmodel.compile()
```

The model must be **compiled** before it can be used. In the next section, we will describe how to configure the 

Like any  random variable object, a probabilistic model is equipped with methods such as *log_prob()* and *sample()*. Then, we can sample data from the model anbd compute the log-likelihood of a data set:
```python
data = probmodel.sample(size = 1000)
log_like = probmodel.log_prob(data)
```

Folowing Edward's approach, a random variable $x$ is associated to a tensor $x^*$ in the computational graph handled by TensorFlow, where the computations takes place. This tensor $x^*$ contains the samples of the random variable $x$, i.e. $x^*\sim p(x|\theta)$. In this way, random variables can be involved in expressive deterministic operations. For example, the following piece of code corresponds to a zero inflated linear regression model 

```python

#Prior
w = Normal(0, 1, dim=d)
w0 = Normal(0, 1)
p = Beta(1,1)

#Likelihood model
with inf.replicate(size = 1000):
    x = Normal(0,1000, dim=d, observed = true)
    h = Binomial(p)
    y0 = Normal(w0 + inf.matmul(x,w, transpose_b = true ), 1),
    y1 = Delta(0.0)
    y = Deterministic(h*y0 + (1-h)*y1, observed = true)

probmodel = ProbModel(vars = [w,w0,p,x,h,y0,y1,y]) 
probmodel.compile()
data = probmodel.sample(size = 1000)
probmodel.fit(data)
```

A special case, it is the inclusion of deep neural networks within our probabilistic model to capture complex non-linear dependencies between the random variables. This is extensively treated in the the Guide to Bayesian Deep Learning. 

Finally, a probablistic model have the following methods:

- ```probmodel.summary()```: prints a summary representation of the model. 
- ```probmodel.get_config()```: returns a dictionary containing the configuration of the model. The model can be reinstantiated from its config via:

```python 
config = probmodel.get_config()
probmodel = ProbModel.from_config(config)
```
- ```model.to_json()```: returns a representation of the model as a JSON string. Note that the representation does not include the weights, only the architecture. You can reinstantiate the same model (with reinitialized weights) from the JSON string via:
```python
from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

----

## Guide to Approximate Inference in Probabilistic Models

The Inference API defines the set of algorithms and methods used to perform inference in a probabilistic model $p(x,z,\theta)$ (where $x$ are the observations, $z$ the local hidden varaibles, and $\theta$ the global parameters of the model). More precesily, the inference problem redues to compute the posterior probability over the latent variables given a data sample $p(z,\theta|x_{train}), because by looking at these posteriors we can uncover the hidden structure in the data. For the running example, $p(mu|x_{train})$ tells us where the centroids of the data while $p(z_n|x_{train}$ shows us to which centroid belongs every data point. 

INFERPY inherits Edward's approach an consider approximate inference solutions, 

$$ q(z,\theta) \approx p(z,\theta | x_{train})$$, 

in which the task is to approximate the posterior $p(z,\theta | x_{train})$ using a family of distritions, $q(z,\theta; \labmda)$, indexed by parameters $\lambda$. 

A probabilistic model in INFERPY should be compiled before we can access these posteriors,

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
 probmodel.compile(infMethod = 'KLqp')   
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

The compilation process allows to choose the inference algorithm through the 'infMethod' argument. In the above example we use 'Klqp', **black box variational inference**. Other inference algorithms include: 'NUTS', 'MCMC', 'KLpq', etc. Look at ? for a detailed description of the available inference algorithms. 

Following INFERPY guiding principles, users can further configure the inference algorithm. 

First, they can define they family 'Q' of approximating distributions, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = mu, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = mu, initializer='ones')
  
 probmodel.compile(infMethod = 'KLqp', Q = [q_mu, q_sigma, q_z_n])
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

By default, the posterior **q** belongs to the same distribution family than **p** , but in the above example we show how we can change that (e.g. we set the posterior over **mu** to obtain a point mass estimate instead of the Gaussian approximation obatined by default). We can also configure how these **q's** are initialized using any of the Keras's initializers. 

Inspired by Keras semantics, we can furhter configuration of the inference algorithm, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = mu, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = mu, initializer='ones')
 
 sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
 infkl_qp = inf.inference.KLqp(optimizer = sgd, dataSize = N)
 probmodel.compile(infMethod = infkl_qp, Q = [q_mu, q_sigma, q_z_n])

 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

Have a look at Inference Zoo to explore other configuration options. 

In the last part of this guide, we highlight that INFERPY directly builds on top of Edward's compositionality idea to design complex infererence algorithms. 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = mu, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = mu, initializer='ones')
 
 infkl_qp = inf.inference.KLqp(optimizer = 'sgd', Q = [q_z_n], innerIter = 10)
 infMAP = inf.inference.MAP(optimizer = 'sgd', Q = [q_mu, q_sigma])

 probmodel.compile(infMethod = [infkl_qp,infMAP])
 
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

With the above sintaxis, we perform a variational EM algorithm, where the E step is repeated 10 times for every MAP step.

More flexibility is also available by defining how each mini batch is process by the inference algorithm. The following piece code is equivalent to the above one, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 

 q_z_n = inf.inference.Q.Multinomial(bind = mu, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = mu, initializer='ones')
 
 infkl_qp = inf.inference.KLqp(optimizer = 'sgd', Q = [q_z_n])
 infMAP = inf.inference.MAP(optimizer = 'sgd', Q = [q_mu, q_sigma])

 emAlg = lambda (infMethod, dataBatch):
    for _ in range(10)
        infMethod[0].update(data = dataBatch)
    
    infMethod[1].update(data = dataBatch)
    return 
 
 probmodel.compile(infMethod = [infkl_qp,infMAP], ingAlg = emAlg)
 
 model.fit(x_train, EPOCHS = 10)
 posterior_mu = probmodel.posterior(mu)
```

Have a look again at Inference Zoo to explore other complex compositional options. 

----

## Guide to Compositional Inference 

Talk about inference as optimization, inference.update as a gradient step, etc... Distuinguish from MCMC, etc... 

----

## Guide to Bayesian Deep Learning

INFERPY inherits Edward's approach for representing probabilistic models as (stochastic) computational graphs. As describe above, a random variable $x$ is associated to a tensor $x^*$ in the computational graph handled by TensorFlow, where the computations takes place. This tensor $x^*$ contains the samples of the random variable $x$, i.e. $x^* \sim p(x|\theta)$. In this way, random variables can be involved in complex deterministic operations containing deep neural networks, math operations and another libraries compatible with Tensorflow (such as Keras).

Bayesian deep learning or deep probabilistic programming enbraces the idea of employing deep neural networks within a probabilistic model in order to capture complex non-linear dependencies between the variables. 

InferPy's API gives support to this powerful and flexible modelling framework. Let us start by showing how a variational autoencoder over binary data can be defined by mixing Kearas and InferPy code. 

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

M = 1000
dim_z = 10
dim_x = 100

#Define the decoder network
input_z  = keras.layers.Input(input_dim = dim_z)
layer = keras.layers.Dense(256, activation = 'relu')(input_z)
output_x = keras.layers.Dense(dim_x)(layer)
decoder_nn = keras.models.Model(inputs = input, outputs = output_x)

#define the generative model
with inf.replicate(size = N)
 z = Normal(0,1, dim = dim_z)
 x = Bernoulli(logits = decoder_nn(z.value()), observed = true)

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
probmodel.fit(x_train)

#extract the hidden representation from a set of observations
hidden_encoding = probmodel.predict(x_pred, targetvar = z)
```

In this case, the parameters of the encoder and decoder neural networks are automatically managed by Keras. These parameters are them treated as model parameters and not exposed to the user. In consequence, we can not be Bayesian about them by defining specific prior distributions. In this example (?) , we show how we can avoid that by introducing extra complexity in the code. 

Other examples of probabilisitc models using deep neural networks are:
 - Bayesian Neural Networks
 - Mixture Density Networks
 - ...

----
## Guide to Validation of Probabilistic Models

Model validation try to assess how faifhfully the inferered probabilistic model represents and explain the observed data. 

The main tool for model validation consists on analyzing the posterior predictive distribution, 

$$ p(y_{test}, x_{test}|y_{train}, x_{train}) = \int p(y_{test}, x_{test}|z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta $$.

This posterior predictive distribution can be used to measure how well the model fits an independent dataset using the test marginallog-likelihood, $\ln p(y_{test}, x_{test}|y_{train}, x_{train})$,  

```python
log_like = probmodel.evaluate(test_data, metrics = ['log_likelihood'])
```

In other cases, we may need to evalute the predictive capacity of the model with respect to some target variable $y$, 

$$ p(y_{test}|x_{test}, y_{train}, x_{train}) = \int p(y_{test}|x_{test},z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta $$,

So the metrics can be computed with respect to this target variable by using the 'targetvar' argument, 

```python
log_like, accuracy, mse = probmodel.evaluate(test_data, targetvar = y, metrics = ['log_likelihood', 'accuracy', 'mse'])
```
So, the log-likelihood metric as well as the accuracy and the mean square error metric are computed by using the predictive posterior $p(y_{test}|x_{test}, y_{train}, x_{train})$. 


Custom evaluation metrics can also be defined, 

```python
def mean_absolute_error(posterior, observations, weights=None):
    predictions = tf.map_fn(lambda x : x.getMean(), posterior)
    
    return tf.metrics.mean_absolute_error(observations, predictions, weights)
    

mse, mae = probmodel.evaluate(test_data, targetvar = y, metrics = ['mse', mean_absolute_error])
```

----

## Guide to Data Handling


----

# Probabilistic Model Zoo





