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
   :scale: 35 %
   :align: center
   
   Bayesian PCA
	
    The **prior model** is the variable :math:`\mu`. The **data model** is the part of the model surrounded by the box indexed by **N**.


And this is how this Bayesian PCA model is denfined in InferPy:

.. literalinclude:: ../../examples/docs/guidemodels/1.py
   :language: python



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
the 'dim' argument (inspired by PyMC3 and Keras):


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python
   :lines: 1-14


The ``with inf.replicate(size = N)`` sintaxis can also be used to define
multi-dimensional objects:



.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python
   :lines: 19-20


Following Edward's approach, the multivariate dimension is the innermost (right-most)
dimension of the parameters.


The argument ``observed = true`` in the constructor of a random variable
is used to indicate whether a variable is observable or not.

Probabilistic Models
--------------------
A **probabilistic model** defines a joint distribution over observable 
and non-observable variables, :math:`p(\theta,\mu,\sigma,z_n, x_n)` for the
running example. The variables in the model are the ones defined using the 
``with inf.ProbModel() as pca:`` construct. Alternatively, we can also use a builder,

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python
   :lines: 24-25

The model must be **compiled** before it can be used.

Like any random variable object, a probabilistic model is equipped with
methods such as  ``sample()``, ``log_prob()`` and  ``sum_log_prob()``. Then, we can sample data
from the model and compute the log-likelihood of a data set:

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python
   :lines: 29-31


Folowing Edward's approach, a random variable :math:`x` is associated to
a tensor :math:`x^*` in the computational graph handled by TensorFlow,
where the computations takes place. This tensor :math:`x^*` contains the
samples of the random variable :math:`x`, i.e.
:math:`x^*\sim p(x|\theta)`. In this way, random variables can be
involved in expressive deterministic operations.


Dependecies between variables are modelled by setting a given variable as a parameter of another variable. For example:

.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 4-9

Moreover, we might consider using the function ``inferpy.case`` as the parameter of other random variables:

.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 12-36



Supported Probability Distributions
-----------------------------------


Supported probability distributions are located in the package ``inferpy.models``. All of them
have ``inferpy.models.RandomVariable`` as superclass. Those currently implemented are:

.. code:: python

   >>> inf.models.ALLOWED_VARS
   ['Bernoulli', 'Beta', 'Categorical', 'Deterministic', 'Dirichlet', 'Exponential', 'Gamma', 'InverseGamma', 'Laplace', 'Multinomial', 'Normal', 'Poisson', 'Uniform']



