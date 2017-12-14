Guide to Building Probabilistic Models
======================================

Getting Started with Probabilistic Models
------------------------------------------

InferPy focuses on *hirearchical probabilistic models* which usually are
structured in two different layers:

-  A **prior model** defining a joint distribution :math:`p(\theta)`
   over the global parameters of the model, :math:`\theta`.
-  A **data or observation model** defining a joint conditional
   distribution :math:`p(x,h|\theta)` over the observed quantities
   :math:`x` and the the local hidden variables :math:`h` governing the
   observation :math:`x`. This data model should be specified in a
   single-sample basis. There are many models of interest without local
   hidden variables, in that case we simply specify the conditional
   :math:`p(x|\theta)`. More flexible ways of defining the data model
   can be found in ?.

A Bayesian PCA model has the following graphical structure, 

.. figure:: ../_static/imgs/LinearFactor.png
   :alt: Linear Factor Model
   :scale: 50 %
   :align: center
   
   Bayesian PCA
	
    The **prior model** is the variable :math:`\mu`. The **data model** is the part of the model surrounded by the box indexed by **N**.


And this is how this Bayesian PCA model is denfined in InferPy:

.. code:: python

    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet
	
    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet

    # Define the probabilistic model
    with inf.ProbModel() as pca:
       # K defines the number of components. 
       K=10
    
       #Prior for the principal components
       with inf.replicate(size = K)
          mu = Normal(loc = 0, scale = 1, dime = d)

       # Number of observations
       N = 1000
    
       #data Model
       with inf.replicate(size = N):
          # Latent representation of the sample
          w_n = Normal(loc = 0, scale = 1, dim = K)
          # Observed sample. The dimensionality of mu is [K,d]. 
          x = Normal(loc = inf.matmul(w_n,mu), scale = 1.0, observed = true)

       #compile the probabilistic model
       pca.compile()

The ``with inf.replicate(size = N)`` sintaxis is used to replicate the
random variables contained within this construct. It follows from the
so-called *plateau notation* to define the data generation part of a
probabilistic model. Every replicated variable is **conditionally
idependent** given the previous random variables (if any) defined
outside the **with** statement.

.. Internally, ``with inf.replicate(size = N)`` construct modifies the
   random variable shape by adding an extra dimension. For the above
   example, z\_n's shape is [N,1], and x\_n's shape is [N,d].


Random Variables
----------------

Following Edward's approach, a random variable :math:`x` is an object
parametrized by a tensor :math:`\theta` (i.e. a TensorFlow's tensor or
numpy's ndarray). The number of random variables in one object is
determined by the dimensions of its parameters (like in Edward) or by
the 'shape' or 'dim' argument (inspired by PyMC3 and Keras):

.. code:: python

    # matrix of [1, 5] univariate standard normals
    x  = Normal(loc = 0, scale = 1, dim = 5) 

    # matrix of [1, 5] univariate standard normals
    x  = Normal(loc = np.zeros(5), scale = np.ones(5)) 

    # matrix of [1,5] univariate standard normals
    x = Normal (loc = 0, scale = 1, shape = [1,5])

The ``with inf.replicate(size = N)`` sintaxis can also be used to define
multi-dimensional objects:

.. code:: python

    # matrix of [10,5] univariate standard normals
    with inf.replicate(size = 10)
        x = Normal (loc = 0, scale = 1, dim = 5)

.. More detailed inforamtion about the semantics of ``with inf.replicate(size = N)`` can be found in ?. Examples of using this construct to define more expressive and complex models can be found in ?.

Multivariate distributions can be defined similarly. Following Edward's
approach, the multivariate dimension is the innermost (right-most)
dimension of the parameters.

.. code:: python

    # Object with five K-dimensional multivariate normals, shape(x) = [5,K]
    x  = MultivariateNormal(loc = np.zeros([5,K]), scale = np.ones([5,K,K])) 

    # Object with five K-dimensional multivariate normals, shape(x) = [5,K]
    x = MultivariateNormal (loc = np.zeros(K), scale = np.ones([K,K]), shape = [5,K])

The argument ``observed = true`` in the constructor of a random variable
is used to indicate whether a variable is observable or not.

Probabilistic Models
--------------------
A **probabilistic model** defines a joint distribution over observable 
and non-observable variables, :math:`p(\theta,\mu,\sigma,z_n, x_n)` for the
running example. The variables in the model are the ones defined using the 
``with inf.ProbModel() as pca:`` construct. Alternatively, we can also use a builder,

.. code:: python

    from inferpy import ProbModel
    pca = ProbModel(vars = [mu,w_n,x_n]) 
    pca.compile()

The model must be **compiled** before it can be used.

Like any random variable object, a probabilistic model is equipped with
methods such as ``log_prob()`` and ``sample()``. Then, we can sample data
from the model and compute the log-likelihood of a data set:

.. code:: python

    data = probmodel.sample(size = 1000)
    log_like = probmodel.log_prob(data)

Folowing Edward's approach, a random variable :math:`x` is associated to
a tensor :math:`x^*` in the computational graph handled by TensorFlow,
where the computations takes place. This tensor :math:`x^*` contains the
samples of the random variable :math:`x`, i.e.
:math:`x^*\sim p(x|\theta)`. In this way, random variables can be
involved in expressive deterministic operations. For example, the
following piece of code corresponds to a zero inflated linear regression
model

.. code:: python


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

A special case, it is the inclusion of deep neural networks within our
probabilistic model to capture complex non-linear dependencies between
the random variables. This is extensively treated in the the Guide to
Bayesian Deep Learning.

Finally, a probablistic model have the following methods:

-  ``probmodel.summary()``: prints a summary representation of the
   model.
-  ``probmodel.get_config()``: returns a dictionary containing the
   configuration of the model. The model can be reinstantiated from its
   config via:

.. code:: python

    config = probmodel.get_config()
    probmodel = ProbModel.from_config(config)

-  ``model.to_json()``: returns a representation of the model as a JSON
   string. Note that the representation does not include the weights,
   only the architecture. You can reinstantiate the same model (with
   reinitialized weights) from the JSON string via: \`\`\`python from
   models import model\_from\_json

.. code:: python
    
    json_string = model.to_json() 
    model = model_from_json(json_string)

Supported Probability Distributions
-----------------------------------

