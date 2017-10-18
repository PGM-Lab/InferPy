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
import inferpy as inf
from inferpy import ProbModel
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
with inf.dataModel():
    # Sample the component of the mixture this sample belongs to. 
    # This is a latent variable that can not be observed
    h = Multinomial(probs = p)
    # Sample the observed value from the Gaussian of the selected component.  
    y = Normal(loc = inf.gather(mu,h), scale = inf.gather(sigma,h), observed = true)
```

The **data model** must be surrounded by a **with** statement to inidicate that the defined random variables will be reapeatedly used in each data sample.

Once the prior and data model are defined, the probablitic model itself can be created and compiled:
```python
probmodel = ProbModel(prior = [p,mu,sigma], dataModel = [h, y]) 
probmodel.compile(infMethod = 'KLqp', optimizer='sgd')
```
During the model compilation we specify the inference method that will be used to learn the model as well as the optimization algorithm. Both the inference method and the optimizer can be further configure. But, as in Keras, a core principle is to try make things reasonbly simple, while allowing the user full control if needed. 

```python
from keras.optimizers import SGD
probmodel = ProbModel(prior = [p,mu,sigma], dataModel = [h, y]) 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
probmodel.compile(infMethod = 'KLqp', optimizer=sgd)
```

Now we can sample data from the modeled if needed:
```python
data = inf.sampleData(size = 1000)
```
Or you can fit your model with a given data set:
```python
probmodel.fit(data_training, epochs=10)
```
or update your probablistic model with new data usian Bayes' rule:
```python
probmodel.update(new_data_training)
```
Query the posterior over a given random varible:
```python
x_post = inf.posterior(mu)
```
Evaluate your model according to a given metric:
```python
metrics = probmodel.evaluate(test_data, metrics = ['log_likelihood', 'mean_squared_error'], targetvar = x)
```
Or compute predicitons on new data
```python
cluster_assignments = model.predict(test_data, targetvar = h)
```
