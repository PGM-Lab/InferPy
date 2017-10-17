# INFERPY: The Python Library for Probabilistic Modelling

INFERPY is a high-level API for probabilistic modelling written in Python and capable of running on top of Edward, Tensorflow and Apache Spark. INFERPY's API is strongly inspired by Keras and it has a focus on enabling flexible data processing, simple probablistic modelling, scalable inference and robust model validation. 

Use INFERPY is you need a probabilistic programming language that:
 - Has a simple and user friendly API (inspired by Keras).
 - Allows for easy and fast prototyping of simple probabilistic models or complex probabilistics constructs containing deep neural networks (by relying on Edward).   
 - Run seamlessly on CPU and GPU (by relying on Tensorflow). 
 - Process seamlessly small data sets or large distributed data sets (by relying on Apache Spark). . 

# Getting Started: 30 seconds to INFERPY 

The core data structure of INFERPY is a **model**, a way to organize the random variables and the conditional dependencies of a probabilistic model. A probabilistic model is organized in two main parts the **prior model** and the **data model**.  

This is how you define the **prior** for the a simple **mixture of Gaussians** model:

```python
import numpy as np
from inferpy.models import Normal, InverseGamma, Dirichlet

# K defines the number of components. 
K=10
#Prior for the means of the Gaussians 
mu = Normal(loc = 0, scale =1, shape=[K,d])
#Prior for the precision of the Gaussians 
sigma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
#Prior for the mixing proportions
p = Dirichlet(np.ones(K))
```
Then the **data model** is defined by showing how to draw a single example:

```python
#data Model
with bs.dataModel():
    # Sample the component of the mixture this sample belongs to. 
    # This is a latent variable that can not be observed
    h = Multinomial(probs = p)
    # Sample the observed value from the Gaussian of the selected component.  
    y = Normal(loc = bs.gather(mu,h), scale = bs.gather(sigma,h), observed = true)
```

The **data model** must be surrounded by a **with** statement to inidicate that the defined random variables will be reapeatedly used in each data sample.

Once the prior and data model are defined, the probablitic model itself can be created and compiled:
```python
probmodel = ProbModel(prior = [p,mu,sigma], dataModel = [h, y]) 
probmodel.compile(optimizer='sgd', infMethod = 'KLqp')
```
