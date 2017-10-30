# InferPy: A Python Library for Probabilistic Modelling

InferPy is a high-level API for probabilistic modelling written in Python and capable of running on top of Edward, Tensorflow and Apache Spark. InferPy's API is strongly inspired by Keras and it has a focus on enabling flexible data processing, simple probablistic modelling, scalable inference and robust model validation. 

Use InferPy is you need a probabilistic programming language that:
 - Has a simple and user friendly API (inspired by Keras).
 - Allows for easy and fast prototyping of simple probabilistic models or complex probabilistics constructs containing deep neural networks (by relying on Edward).   
 - Run seamlessly on CPU and GPU (by relying on Tensorflow). 
 - Process seamlessly small data sets or large distributed data sets (by relying on Apache Spark). . 

----

## Getting Started: 30 seconds to InferPy 

The core data structures of InferPy is a a **probabilistic model**, defined as a set of **random variables** with a conditional independence structure. Like in Edward, a **random varible** is an object parameterized by a set of tensors. 

Let's look at a simple examle. We start defining the **prior** of the parameters of a **mixture of Gaussians** model: 


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

InferPy supports the definition of **plateau notation** by using the construct ```with inf.replicate(size = N) ```, which replicates N times the random variables enclosed within this anotator. This is usefuel when defining the model for the data:

```python
# Number of observations
N = 1000
#data Model
with inf.replicate(size = N)
    # Sample the component indicator of the mixture. This is a latent variable that can not be observed
    z_n = Multinomial(probs = p)
    # Sample the observed value from the Gaussian of the selected component.  
    x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(sigma,z_n), observed = true)
```
As commented above, the variable z_n and x_n are surrounded by a **with** statement to inidicate that the defined random variables will be reapeatedly used in each data sample.

Once the random variables of the  model are defined, the probablitic model itself can be created and compiled. The probabilistic model defines a joint probability distribuiton over all these random variables.  
```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
probmodel.compile(infMethod = 'KLqp'')
```
During the model compilation we specify different inference methods that will be used to learn the model. 

```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
probmodel.compile(infMethod = 'MCMC')
```

The inference method can be further configure. But, as in Keras, a core principle is to try make things reasonbly simple, while allowing the user the full control if needed. 

```python
from keras.optimizers import SGD
probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
infklqp = inf.inference.KLqp(optimizer = sgd, dataSize = N)
probmodel.compile(infMethod = infklqp)
```

Every random variable object is equipped with methods such as *log_prob()* and *sample()*. Similarly, a probabilistic model is also equipped with the same methods. Then, we can sample data from the model anbd compute the log-likelihood of a data set:
```python
data = probmodel.sample(vars = [x_n], size = 100)
log_like = probmodel.log_prob(data)
```

Of course, you can fit your model with a given data set:
```python
probmodel.fit(data_training, epochs=10)
```
Update your probablistic model with new data using the Bayes' rule:
```python
probmodel.update(new_data)
```
Query the posterior over a given random varible:
```python
mu_post = probmodel.posterior(mu)
```
Evaluate your model according to a given metric:
```python
log_like = probmodel.evaluate(test_data, metrics = ['log_likelihood'])
```
Or compute predicitons on new data
```python
cluster_assignments = probmodel.predict(test_data, targetvar = z_n)
```

----

## Guiding Principles

- InferPy's probability distribuions are mainly inherited from TensorFlow Distribuitons package. InferPy's API is fully compatible with tf.distributions' API. The 'shape' argument was added as a simplifing option when defining multidimensional distributions. 
- InferPy directly relies on top of Edward's inference engine and includes all the inference algorithms included in this package. As Edward's inference engine relies on TensorFlow computing engine, InferPy also relies on it too.  
- InferPy seamsly process data contained in a numpy array, Tensorflow's tensor, Tensorflow's Dataset (tf.Data API) or Apache Spark's DataFrame. 
- InferPy also includes novel distributed statistical inference algorithms by combining Tensorflow and Apache Spark computing engines. 

----

## Getting Started
### Guide to Building Probabilistic Models

InferPy focuses on *hirearchical probabilistic models* which usually are structured in two different layers:

- A **prior model** defining a joint distribution $p(\theta)$ over the global parameters of the model, $\theta$.  
- A **data or observation model** defining a joint conditional distribution $p(x,h|\theta)$ over the observed quantities $x$ and the the local hidden variables $h$ governing the observation $x$. This data model should be specified in a single-sample basis. There are many models of interest without local hidden variables, in that case we simply specify the conditional $p(x|\theta)$. More flexible ways of defining the data model can be found in ?. 

This is how a mixture of Gaussians models is denfined in InferPy: 
```python
import numpy as np
import inferpy as inf
from inferpy.models import Normal, InverseGamma, Dirichlet

# K defines the number of components. 
K=10
#Prior for the means of the Gaussians 
mu = Normal(loc = 0, scale = 1, shape=[K,d])
#Prior for the precision of the Gaussians 
invgamma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
#Prior for the mixing proportions
theta = Dirichlet(np.ones(K))

# Number of observations
N = 1000
#data Model
with inf.replicate(size = N)
    # Sample the component indicator of the mixture. This is a latent variable that can not be observed
    z_n = Multinomial(probs = theta)
    # Sample the observed value from the Gaussian of the selected component.  
    x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(invgamma,z_n), observed = true)

#Probabilistic Model
probmodel = ProbModel(prior = [p,mu,sigma,z_n,x_n]) 
probmodel.compile()
```

The ```with inf.replicate(size = N)``` sintaxis is used to replicate the random variables contained within this construct. It follows from the so-called *plateau notation* to define the data generation part of a probabilistic model. Internally, ```with inf.replicate(size = N)``` construct modifies the random variable shape by adding an extra dimension. For the above example, z_n's shape is [N,1], and x_n's shape is [N,d].  

Following Edward's approach, a random variable $x$ is an object parametrized by a tensor $\theta$ (i.e. a TensorFlow's tensor or numpy's ndarray). The number of random variables in one object is determined by the dimensions of its parameters (like in Edward) or by the 'shape' or 'dim' argument (inspired by PyMC3 and Keras):

```python
# vector of 5 univariate standard normals
x  = Normal(loc = 0, scale = 1, dim = 5) 

# vector of 5 univariate standard normals
x  = Normal(loc = np.zeros(5), scale = np.ones(5)) 

# vector of 5 univariate standard normals
x = Normal (loc = 0, scale = 1, shape = [5,1])
```

The ```with inf.replicate(size = N)``` sintaxis can also be used to define multi-dimensional objects, the following code is also equivalent to the above ones:

```python
# vector of 5 univariate standard normals
with inf.replicate(size = 5)
    x = Normal (loc = 0, scale = 1)
```

More detailed inforamtion about the semantics of ```with inf.replicate(size = N)``` can be found in ?. Examples of using this construct to define more expressive and complex models can be found in ?. 


Multivariate distributions can be defined similarly. Following Edward's approach, the multivariate dimension is the innermost (right-most) dimension of the parameters. 
```python
# 2 x 3 matrix of K-dimensional multivariate normals
x  = MultivariateNormal(loc = np.zeros([2,3,K]), scale = np.ones([2,3,K,K]), observed = true) 

# 2 x 3 matrix of K-dimensional multivariate normals
y = MultivariateNormal (loc = np.zeros(K), scale = np.ones([K,K]), shape = [2,3], observed = true)
```

The argument **observed = true** in the constructor of a random variable is used to indicate whether a variable is observable or not.  

A **probabilistic model** defines a joint distribution over observable and non-observable variables, $p(theta,mu,sigma,z_n, x_n)$ for the running example, 

```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
probmodel.compile()
```

The model must be **compiled** before it can be used.

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
    y0 = Normal(w0 + inf.matmul(x,w, transpose_b = true), 1),
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

The API defines the set of algorithms and methods used to perform inference in a probabilistic model $p(x,z,\theta)$ (where $x$ are the observations, $z$ the local hidden variibles, and $\theta$ the global parameters of the model). More precisely, the inference problem reduces to compute the posterior probability over the latent variables given a data sample $p(z,\theta|x_{train}), because by looking at these posteriors we can uncover the hidden structure in the data. For the running example, $p(mu|x_{train})$ tells us where the centroids of the data are, while $p(z_n|x_{train})$ shows us to which centroid every data point belongs to. 

InferPy inherits Edward's approach an consider approximate inference solutions, 

$$ q(z,\theta) \approx p(z,\theta | x_{train})$$, 

in which the task is to approximate the posterior $p(z,\theta | x_{train})$ using a family of distributions, $q(z,\theta; \labmda)$, indexed by a parameter vector $\lambda$. 

A probabilistic model in InferPy should be compiled before we can access these posteriors,

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
 probmodel.compile(infMethod = 'KLqp')   
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

The compilation process allows to choose the inference algorithm through the 'infMethod' argument. In the above example we use 'Klqp'. Other inference algorithms include: 'NUTS', 'MCMC', 'KLpq', etc. Look at ? for a detailed description of the available inference algorithms. 

Following InferPy guiding principles, users can further configure the inference algorithm. 

First, they can define they family 'Q' of approximating distributions, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
  
 probmodel.compile(infMethod = 'KLqp', Q = [q_mu, q_sigma, q_z_n])
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

By default, the posterior **q** belongs to the same distribution family than **p** , but in the above example we show how we can change that (e.g. we set the posterior over **mu** to obtain a point mass estimate instead of the Gaussian approximation used by default). We can also configure how these **q's** are initialized using any of the Keras's initializers. 

Inspired by Keras semantics, we can furhter configure the inference algorithm, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
 
 sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
 infkl_qp = inf.inference.KLqp(optimizer = sgd, dataSize = N)
 probmodel.compile(infMethod = infkl_qp, Q = [q_mu, q_sigma, q_z_n])

 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

Have a look at Inference Zoo to explore other configuration options. 

In the last part of this guide, we highlight that InferPy directly builds on top of Edward's compositionality idea to design complex infererence algorithms. 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
 
 q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
 
 infkl_qp = inf.inference.KLqp(optimizer = 'sgd', Q = [q_z_n], innerIter = 10)
 infMAP = inf.inference.MAP(optimizer = 'sgd', Q = [q_mu, q_sigma])

 probmodel.compile(infMethod = [infkl_qp,infMAP])
 
 model.fit(x_train)
 posterior_mu = probmodel.posterior(mu)
```

With the above sintaxis, we perform a variational EM algorithm, where the E step is repeated 10 times for every MAP step.

More flexibility is also available by defining how each mini-batch is processed by the inference algorithm. The following piece of code is equivalent to the above one, 

```python
 probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 

 q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
 q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
 q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
 
 infkl_qp = inf.inference.KLqp(Q = [q_z_n])
 infMAP = inf.inference.MAP(Q = [q_mu, q_sigma])

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

## Guide to Bayesian Deep Learning

InferPy inherits Edward's approach for representing probabilistic models as (stochastic) computational graphs. As describe above, a random variable $x$ is associated to a tensor $x^*$ in the computational graph handled by TensorFlow, where the computations takes place. This tensor $x^*$ contains the samples of the random variable $x$, i.e. $x^* \sim p(x|\theta)$. In this way, random variables can be involved in complex deterministic operations containing deep neural networks, math operations and another libraries compatible with Tensorflow (such as Keras).

Bayesian deep learning or deep probabilistic programming enbraces the idea of employing deep neural networks within a probabilistic model in order to capture complex non-linear dependencies between the variables. 

InferPy's API gives support to this powerful and flexible modelling framework. Let us start by showing how a variational autoencoder over binary data can be defined by mixing Keras and InferPy code. 

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

## Bayesian Linear Regression

```python
# Shape = [1,d]
w = Normal(0, 1, dim=d)
# Shape = [1,1]
w0 = Normal(0, 1)

with inf.replicate(size = N):
    # Shape = [N,d]
    x = Normal(0,1, dim=d, observed = true)
    # Shape = [1,1] + [N,d]@[d,1] = [1,1] + [N,1] = [N,1] (by broadcasting)
    y = Normal(w0 + tf.matmul(x,w, transpose_b = true ), 1, observed = true)

model = ProbModel(vars = [w0,w,x,y]) 

data = model.sample(size=N)

log_prob = model.log_prob(sample)

model.compile(infMethod = 'KLqp')

model.fit(data)

print(probmodel.posterior([w0,w]))
```
---

## Zero Inflated Linear Regression

```python
# Shape = [1,d]
w = Normal(0, 1, dim=d)
# Shape = [1,1]
w0 = Normal(0, 1)

# Shape = [1,1]
p = Beta(1,1)

with inf.replicate(size = N):
    # Shape [N,d]
    x = Normal(0,1000, dim=d, observed = true)
    # Shape [N,1]
    h = Binomial(p)
    # Shape [1,1] + [N,d]@[d,1] = [1,1] + [N,1] = [N,1] (by broadcasting)
    y0 = Normal(w0 + inf.matmul(x,w, transpose_b = true ), 1),
    # Shape [N,1]
    y1 = Delta(0.0)
    # Shape [N,1]*[N,1] + [N,1]*[N,1] = [N,1]
    y = Deterministic(h*y0 + (1-h)*y1, observed = true)

model = ProbModel(vars = [w0,w,p,x,h,y0,y1,y]) 

data = model.sample(size=N)

log_prob = model.log_prob(sample)

model.compile(infMethod = 'KLqp')

model.fit(data)

print(probmodel.posterior([w0,w]))
```
---

## Bayesian Logistic Regression

```python
# Shape = [1,d]
w = Normal(0, 1, dim=d)
# Shape = [1,1]
w0 = Normal(0, 1)

with inf.replicate(size = N):
    # Shape = [N,d]
    x = Normal(0,1, dim=d, observed = true)
    # Shape = [1,1] + [N,d]@[d,1] = [1,1] + [N,1] = [N,1] (by broadcasting)
    y = Binomial(logits = w0 + tf.matmul(x,w, transpose_b = true), observed = true)

model = ProbModel(vars = [w0,w,x,y]) 

data = model.sample(size=N)

log_prob = model.log_prob(sample)

model.compile(infMethod = 'KLqp')

model.fit(data)

print(probmodel.posterior([w0,w]))
```
---

## Bayesian Multinomial Logistic Regression

```python
# Number of classes
K=10

with inf.replicate(size = K):
    # Shape = [K,d]
    w = Normal(0, 1, dim=d)
    # Shape = [K,1]
    w0 = Normal(0, 1])

with inf.replicate(size = N):
    # Shape = [N,d]
    x = Normal(0,1, dim=d, observed = true)
    # Shape = [1,K] + [N,d]@[d,K] = [1,K] + [N,K] = [N,K] (by broadcasting)
    y = Multinmial(logits = tf.transpose(w0) + tf.matmul(x,w, transpose_b = true), observed = true)

model = ProbModel(vars = [w0,w,x,y]) 

data = model.sample(size=N)

log_prob = model.log_prob(sample)

model.compile(infMethod = 'KLqp')

model.fit(data)

print(probmodel.posterior([w0,w]))
```

---


## Mixture of Gaussians

![Mixture of Gaussians](https://github.com/amidst/InferPy/blob/master/docs/imgs/MoG.png)

Version A 
```python
d=3
K=10
N=1000
#Prior
with inf.replicate(size = K):
    #Shape [K,d]
    mu = Normal(loc = 0, scale =1, dim=d)
    #Shape [K,d]
    sigma = InverseGamma(concentration = 1, rate = 1, dim=d)

# Shape [1,K]
p = Dirichlet(np.ones(K))

#Data Model
with inf.replicate(size = N):
    # Shape [N,1]
    z_n = Multinomial(probs = p)
    # Shape [N,d]
    x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(sigma,z_n), observed = true)
    
model = ProbModel(vars = [p,mu,sigma,z_n, x_n]) 

data = model.sample(size=N)

log_prob = model.log_prob(sample)

model.compile(infMethod = 'KLqp')

model.fit(data)

print(probmodel.posterior([mu,sigma]))

```

Version B
```python
d=3
K=10
N=1000
#Prior
mu = Normal(loc = 0, scale =1, shape = [K,d])
sigma = InverseGamma(concentration = 1, rate = 1, shape = [K,d])

# Shape [1,K]
p = Dirichlet(np.ones(K))

#Data Model
z_n = Multinomial(probs = p, shape = [N,1])
# Shape [N,d]
x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(sigma,z_n), observed = true)
    
probmodel = ProbModel(vars = [p,mu,sigma,z_n, x_n]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([mu,sigma]))
```

---

## Linear Factor Model (PCA)

![Linear Factor Model](https://github.com/amidst/InferPy/blob/master/docs/imgs/LinearFactor.png)

```python
K = 5
d = 10
N=200

with inf.replicate(size = K)
    # Shape [K,d]
    mu = Normal(0,1, dim = d)

# Shape [1,d]
mu0 = Normal(0,1, dim = d)

sigma = 1.0

with inf.replicate(size = N):
    # Shape [N,K]
    w_n = Normal(0,1, dim = K)
    # inf.matmul(w_n,mu) has shape [N,K] x [K,d] = [N,d] by broadcasting mu. 
    # Shape [1,d] + [N,d] = [N,d] by broadcasting mu0
    x = Normal(mu0 + inf.matmul(w,mu), sigma, observed = true)

probmodel = ProbModel([mu,mu0,w_n,x]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([mu,mu0]))
```

---

## PCA with ARD Prior (PCA)

```python
K = 5
d = 10
N=200

with inf.replicate(size = K)
    # Shape [K,d]
    alpha = InverseGamma(1,1, dim = d)
    # Shape [K,d]
    mu = Normal(0,1, dim = d)

# Shape [1,d]
mu0 = Normal(0,1, dim = d)

# Shape [1,1]
sigma = InverseGamma(1,1, dim = 1)

with inf.replicate(size = N):
    # Shape [N,K]
    w_n = Normal(0,1, dim = K)
    # inf.matmul(w_n,mu) has shape [N,K] x [K,d] = [N,d] by broadcasting mu. 
    # Shape [1,d] + [N,d] = [N,d] by broadcasting mu0
    x = Normal(mu0 + inf.matmul(w,mu), sigma, observed = true)

probmodel = ProbModel([alpha,mu,mu0,sigma,w_n,x]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([alpha,mu,mu0,sigma]))
```

---

## Mixed Membership Model

![Mixed Membership Model](https://github.com/amidst/InferPy/blob/master/docs/imgs/LinearFactor.png)

```python
K = 5
d = 10
N=200
M=50

with inf.replicate(size = K)
    #Shape = [K,d]
    mu = Normal(0,1, dim = d)
    #Shape = [K,d]
    sigma = InverseGamma(1,1, dim = d)

with inf.replicate(size = N):
    #Shape = [N,K]
    theta_n = Dirichlet(np.ones(K))
    with inf.replicate(size = M):
        # Shape [N*M,1]
        z_mn = Multinomial(theta_n)
        # Shape [N*M,d]
        x = Normal(tf.gather(mu,z_mn), tf.gather(sigma,z_mn), observed = true)

probmodel = ProbModel([mu,sigma,theta_n,z_mn,x]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([mu,sigma]))

```
---

## Latent Dirichlet Allocation


```python
K = 5 # Number of topics 
d = 1000 # Size of vocabulary
N=200 # Number of documents in the corpus
M=50 # Number of words in each document

with inf.replicate(size = K)
    #Shape = [K,d]
    dir = Dirichlet(np.ones(d)*0.1)

with inf.replicate(size = N):
    #Shape = [N,K]
    theta_n = Dirichlet(np.ones(K))
    with inf.replicate(size = M):
        # Shape [N*M,1]
        z_mn = Multinomial(theta_n)
        # Shape [N*M,d]
        x = Multinomial(tf.gather(dir,z_mn), tf.gather(dir,z_mn), observed = true)

probmodel = ProbModel([dir,theta_n,z_mn,x]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior(dir))

```

---

## Matrix Factorization

![Matrix Factorization Model](https://github.com/amidst/InferPy/blob/master/docs/imgs/MatrixFactorization.png)

Version A
```python
N=200
M=50
K=5

with inf.replicate(name = 'A', size = M)
    # Shape [M,K]
    gamma_m = Normal(0,1, dim = K)

with inf.replicate(name = 'B', size = N):
    # Shape [N,K]
    w_n = Normal(0,1, dim = K)
    
with inf.replicate(compound = ['A', 'B']):
    # x_mn has shape [N,K] x [K,M] = [N,M]
    x_nm = Normal(tf.matmul(w_n,gamma_m, transpose_b = true), 1, observed = true)


probmodel = ProbModel([w_n,gamma_m,x_nm]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([w_n,gamma_m]))

```

Version B
```python
N=200
M=50
K=5

# Shape [M,K]
gamma_m = Normal(0,1, shape = [M,K])

# Shape [N,K]
w_n = Normal(0,1, shape = [N,K])
    
# x_mn has shape [N,K] x [K,M] = [N,M]
x_nm = Normal(tf.matmul(w_n,gamma_m, transpose_b = true), 1, observed = true)

probmodel = ProbModel([w_n,gamma_m,x_nm]) 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

print(probmodel.posterior([w_n,gamma_m]))

```

---

## Linear Mixed Effect Model 


```python

N = 1000 # number of observations
n_s = 100 # number of students
n_d = 10 # number of instructor
n_dept = 10 # number of departments

eta_s = Normal(0,1, dim = n_s)
eta_d = Normal(0,1, dim = n_d)
eta_dept = Normal(0,1, dim = n_dept)
mu = Normal(0,1)
mu_service = Normal(0,1)

with inf.replicate( size = N):
    student = Multinomial(probs = np.rep(1,n_s)/n_s, observed = true)
    instructor = Multinomial(probs = np.rep(1,n_d)/n_d, observed = true)
    department = Multinomial(probs = np.rep(1,n_dept)/n_dept, observed = true)
    service = Binomial (probs = 0.5, observed = true)
    y = Normal (tf.gather(eta_s,student) 
                + bs.gather(eta_d,instructor) 
                + bs.gather(eta_dept,department) 
                +  mu + mu_service*service, 1, observed = true)

#vars = 'all' automatically add all previously created random variables
probmodel = ProbModel(vars = 'all') 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

#When no argument is given to posterior, return all non-replicated random varibles
print(probmodel.posterior())
```

---

## Bayesian Neural Network Classifier 


```python
d = 10   # number of features
N = 1000 # number of observations

def neural_network(x):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])

W_0 = Normal(0,1, shape = [d,10])
W_1 = Normal(0,1, shape = [10,10])
W_2 = Normal(0,1, shape = [10,1])

b_0 = Normal(0,1, shape = [1,10])
b_1 = Normal(0,1, shape = [1,10])
b_2 = Normal(0,1, shape = [1,1])


with inf.replicate(size = N):
    x = Normal(0,1, dim = d, observed = true)
    y = Bernoulli(logits=neural_network(x), observed = true)

#vars = 'all' automatically add all previously created random variables
probmodel = ProbModel(vars = 'all') 

data = probmodel.sample(size=N)

log_prob = probmodel.log_prob(sample)

probmodel.compile(infMethod = 'KLqp')

probmodel.fit(data)

#When no argument is given to posterior, return all non-replicated random varibles
print(probmodel.posterior())
```

---

## Variational Autoencoder 


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
