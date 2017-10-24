# INFERPY: The Python Library for Probabilistic Modelling

INFERPY is a high-level API for probabilistic modelling written in Python and capable of running on top of Edward, Tensorflow and Apache Spark. INFERPY's API is strongly inspired by Keras and it has a focus on enabling flexible data processing, simple probablistic modelling, scalable inference and robust model validation. 

Use INFERPY is you need a probabilistic programming language that:
 - Has a simple and user friendly API (inspired by Keras).
 - Allows for easy and fast prototyping of simple probabilistic models or complex probabilistics constructs containing deep neural networks (by relying on Edward).   
 - Run seamlessly on CPU and GPU (by relying on Tensorflow). 
 - Process seamlessly small data sets or large distributed data sets (by relying on Apache Spark). . 

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
During the model compilation we specify the inference method that will be used to learn the model. The inference method can be further configure. But, as in Keras, a core principle is to try make things reasonbly simple, while allowing the user full control if needed. 

```python
from keras.optimizers import SGD
probmodel = ProbModel(prior = [p,mu,sigma], dataModel = [h, y]) 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
infklqp = inf.KLqp(optimizer = sgd, dataSize = N)
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

## Guiding Principles

- INFERPY's probability distribuionts are mainly inherited from TensorFlow Distribuitons package. INFERPY's API is fully compatible with tf.distributions' API. The 'shape' argument was added as a simplifing option when defining multidimensional distributions. 
- INFERPY directly relies on top of Edward's inference engine and includes all the inference algorithms included in this package. As Edward's inference engine relies on TensorFlow computing engine, INFERPY also relies on it too.  
- INFERPY seamsly process data contained in a numpy array, Tensorflow's tensor, Tensorflow's Dataset (tf.Data API) or Apache Spark's DataFrame. 
- INFERPY also includes novel distributed statistical inference algorithms by combining Tensorflow and Apache Spark computing engines. 


## Getting Started
### Probabilistic Model

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

Multivariate distributions can be defined similarly. Similiarly to Edward's approach, the multivariate dimension is the innermost (right-most) dimension of the parameters. 
```python
# 2 x 3 matrix of K-dimensional multivariate normals
x  = MultivariateNormal(loc = np.zeros((2,3,K)), scale = np.ones((2,3,K,K))) 

# 2 x 3 matrix of K-dimensional multivariate normals
y = MultivariateNormal (loc = np.zeros(K), scale = np.ones((K,K)), shape = [2,3])
```

The argument **observed = true** in the constructor of a random variable is used to indicate whether a variable is observable or not.  

A **probabilistic model** defines a joint distribution $p(theta,mu,sigma,z_n, x_n)$ over observable and non-observable variables, 
```python
from inferpy import ProbModel
probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
```
Like any  random variable object, a probabilistic model is equipped with methods such as *log_prob()* and *sample()*. Then, we can sample data from the model anbd compute the log-likelihood of a data set:
```python
data = probmodel.sample(size = 1000)
log_like = probmodel.log_prob(data)
```

Folowing Edward's approach, a random variable $x$ is associated to a tensor $x^*$ in the computational graph handled by TensorFlow, where the computations takes place. This tensor *x** contains the samples of the random variable $x$, i.e. $x^*\sim p(x|\theta)$. In this way, random variables can be involved in complex deterministic operations containing deep neural networks, math operations and another libraries compatible with Edward and Tensorflow. 

