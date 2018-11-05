Guide to Building Probabilistic Models
======================================

Getting Started with Probabilistic Models
------------------------------------------

InferPy focuses on *hirearchical probabilistic models* structured 
in two different layers:

-  A **prior model** defining a joint distribution :math:`p(\mathbf{w})`
   over the global parameters of the model. :math:`\mathbf{w}` can be a single random
   variable or a bunch of random variables with any given dependency structure. 
-  A **data or observation model** defining a joint conditional
   distribution :math:`p(\mathbf{x},\mathbf{z}|\mathbf{w})` over the observed quantities
   :math:`\mathbf{x}` and the the local hidden variables :math:`\mathbf{z}` governing the
   observation :math:`\mathbf{x}`. This data model is specified in a
   single-sample basis. There are many models of interest without local
   hidden variables, in that case, we simply specify the conditional
   :math:`p(\mathbf{x}|\mathbf{w})`. Similarly, either :math:`\mathbf{x}` or 
   :math:`\mathbf{z}` can be a single random variable or a bunch of random variables 
   with any given dependency structure.


For example, a Bayesian PCA model has the following graphical structure, 

.. figure:: ../_static/imgs/LinearFactor.png
   :alt: Linear Factor Model
   :scale: 35 %
   :align: center
   
   Bayesian PCA
	
	The **prior model** are the variables :math:`w_k`. The **data model** is the part of the model surrounded by the box indexed by **N**.


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
dimension of the parameters. By contrast, with this replicate construct, we define the number
of batches. In case of 'dim' being a list of 2 elements, the number of batches is specified as well.
For example, the following code is equivalent to the previous one.



.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python
   :lines: 22


Note that indexing is supported:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python
   :lines: 24-30


Moreover, we may use indexation for defining new variables whose indexes may be other (discrete) variables:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python
   :lines: 34-35



Any random variable in InferPy contain the following (optional) input parameters
in the constructor:

- ``validate_args`` : Python boolean indicating that possibly expensive checks with the input parameters are enabled.
  By default, it is set to ``False``.

- ``allow_nan_stats`` : When ``True``, the value "NaN" is used to indicate the result is undefined. Otherwise an exception is raised.
  Its default value is ``True``.

- ``name``: Python string with the name of the underlying Tensor object.

- ``observed``: Python boolean which is used to indicate whether a variable is observable or not . The default value is ``False``

- ``dim``: dimension of the variable. The default value is ``None``


Inferpy supports a wide range of probability distributions. Details of the specific arguments 
for each supported distributions are specified in the following sections.
`



Probabilistic Models
--------------------
A **probabilistic model** defines a joint distribution over observable
and non-observable variables, :math:`p(\mathbf{w}, \mathbf{z}, \mathbf{x})` for the
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


.. Folowing Edward's approach, a random variable :math:`x` is associated to
.. a tensor :math:`x^*` in the computational graph handled by TensorFlow,
.. where the computations takes place. This tensor :math:`x^*` contains the
.. samples of the random variable :math:`x`, i.e.
.. :math:`x^*\sim p(x|\theta)`. 

Random variables can be involved in expressive deterministic operations. Dependecies 
between variables are modelled by setting a given variable as a parameter of another variable. For example:

.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 4-9

Moreover, we might consider using the function ``inferpy.case`` as the parameter of other random variables:

.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 12-36


Note that we might use the case function inside the replicate construct. The result will be
a multi-batch random variable having the same distribution for each batch. When obtaining a sample from
the model, each sample of a given batch in x is independent of the rest.

.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 41-46


We can also use the functions ``inferpy.case_states`` or ``inferpy.gather`` for defining
the same model.


.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 50-62


We can use the function ``inferpy.case_states`` with a list of variables (or multidimensional variables):


.. literalinclude:: ../../examples/docs/guidemodels/4.py
   :language: python
   :lines: 67-95






Supported Probability Distributions
-----------------------------------


Supported probability distributions are located in the package ``inferpy.models``. All of them
have ``inferpy.models.RandomVariable`` as superclass. A list with all the supported distributions can be obtained as
as follows.

.. code:: python

   >>> inf.models.ALLOWED_VARS
   ['Bernoulli', 'Beta', 'Categorical', 'Deterministic', 'Dirichlet', 'Exponential', 'Gamma', 'InverseGamma', 'Laplace', 'Multinomial', 'Normal', 'Poisson', 'Uniform']


Bernoulli
~~~~~~~~~~~~~~~

Binary distribution which takes the value 1 with probability :math:`p` and the value with :math:`1-p`. Its probability mass
function is

.. math::

   p(x;p) =\left\{\begin{array}{cc} p & \mathrm{if\ } x=1 \\
    1-p & \mathrm{if\ } x=0 \\ \end{array} \right.


An example of definition in InferPy of a random variable following a Bernoulli distribution is shown below. Note that the
input parameter ``probs`` corresponds to :math:`p` in the previous equation.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 8-12


This distribution can be initialized by indicating the logit function of the probability, i.e., :math:`logit(p) = log(\frac{p}{1-p})`.


Beta
~~~~~~~~~~~~~~~

Continuous distribution defined in the interval :math:`[0,1]` and parametrized by two positive shape parameters,
denoted :math:`\alpha` and :math:`\beta`.

.. math::

   p(x;\alpha,\beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}

where `B` is the beta function

.. math::

   B(\alpha,\beta)=\int_{0}^{1}t^{\alpha-1}(1-t)^{\beta-1}dt



The definition of a random variable following a Beta distribution is done as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 19-23

Note that the input parameters ``concentration0`` and ``concentration1`` correspond to the shape
parameters :math:`\alpha` and :math:`\beta` respectively.


Categorical
~~~~~~~~~~~~~~~

Discrete probability distribution that can take :math:`k` possible states or categories. The probability
of each state is separately defined:

.. math::

   p(x;\mathbf{p}) = p_i

where :math:`\mathbf{p} = (p_1, p_2, \ldots, p_k)` is a k-dimensional vector with the probability associated to each possible state.





The definition of a random variable following a Categorical distribution is done as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 28-32




Deterministic
~~~~~~~~~~~~~~~


The deterministic distribution is a probability distribution in a space (continuous or discrete) that always takes
the same value :math:`k_0`. Its probability density (or mass) function can be defined as follows.


.. math::

   p(x;k_0) =\left\{\begin{array}{cc} 1 & \mathrm{if\ } x=k_0 \\
    0 & \mathrm{if\ } x \neq k_0 \\ \end{array} \right.




The definition of a random variable following a Beta distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 37


where the input parameter ``loc`` corresponds to the value :math:`k_0`.


Dirichlet
~~~~~~~~~~~~~~~


Dirichlet distribution is a continuous multivariate probability distribution parmeterized by a vector of positive reals
:math:`(\alpha_1,\alpha_2,\ldots,\alpha_k)`.
It is a multivariate generalization of the beta distribution. Dirichlet distributions are commonly used as prior
distributions in Bayesian statistics. The Dirichlet distribution of order :math:`k \geq 2` has the following density function.




.. math::

   p(x_1,x_2,\ldots x_k; \alpha_1,\alpha_2,\ldots,\alpha_k) = {\frac{\Gamma\left(\sum_i \alpha_i\right)}
   {\prod_i \Gamma(\alpha_i)} \prod_{i=1}^k x_i^{\alpha_i-1}}{}





The definition of a random variable following a Beta distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 49-53

where the input parameter ``concentration`` is the vector  :math:`(\alpha_1,\alpha_2,\ldots,\alpha_k)`.




Exponential
~~~~~~~~~~~~~~~

The exponential distribution (also known as negative exponential distribution) is defined over a continuous domain and
describes the time between events in a Poisson point process, i.e., a process in which events occur continuously
and independently at a constant average rate. Its probability density function is



.. math::

   p(x;\lambda) =\left\{\begin{array}{cc} \lambda e^{-\lambda x} & \mathrm{if\ } x\geq 0 \\
    0 & \mathrm{if\ } x < k_0 \\ \end{array} \right.


where :math:`\lambda>0` is the rate or inverse scale.


The definition of a random variable following a exponential distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 57-61

where the input parameter ``rate`` corresponds to the value :math:`\lambda`.




Gamma
~~~~~~~~~~~~~~~


The Gamma distribution is a continuous probability distribution parametrized by a concentration (or shape)
parameter :math:`\alpha>0`, and an inverse scale parameter :math:`\lambda>0` called rate. Its density function is
defined as follows.


.. math::

   p(x;\alpha, \beta) = \frac{\beta^\alpha x^{\alpha - 1} e^{\beta x}}{\Gamma(\alpha)}


for :math:`x > 0` and where :math:`\Gamma(\alpha)` is the gamma function.


The definition of a random variable following a gamma distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 67

where the input parameters ``concentration`` and ``rate`` corespond to  :math:`\alpha` and :math:`\beta` respectively.





Inverse-gamma
~~~~~~~~~~~~~~~


The Inverse-gamma distribution is a continuous probability distribution which is the distribution of the reciprocal
of a variable distributed according to the gamma distribution. It is also parametrized by a concentration (or shape)
parameter :math:`\alpha>0`, and an inverse scale parameter :math:`\lambda>0` called rate. Its density function is
defined as follows.


.. math::

   p(x;\alpha, \beta) = \frac{\beta^\alpha x^{-\alpha - 1} e^{-\frac{\beta}{x}}}{\Gamma(\alpha)}


for :math:`x > 0` and where :math:`\Gamma(\alpha)` is the gamma function.


The definition of a random variable following a inverse-gamma distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 73

where the input parameters ``concentration`` and ``rate`` corespond to  :math:`\alpha` and :math:`\beta` respectively.







Laplace
~~~~~~~~~~~~~~~

The Laplace distribution is a continuous probability distribution with the following density function

.. math::

   p(x;\mu,\sigma) = \frac{1}{2\sigma} exp \left( - \frac{|x - \mu |}{\sigma}\right)




The definition of a random variable following a Beta distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 80-84

where the input parameter ``loc`` and ``scale`` correspond to :math:`\mu` and :math:`\sigma` respectively.





Multinomial
~~~~~~~~~~~~~~~

The multinomial is a discrete distribution which models the probability of counts resulting from repeating :math:`n`
times an experiment with :math:`k` possible outcomes. Its probability mass function is defined below.

.. math::

   p(x_1,x_2,\ldots x_k; \mathbf{p}) =  \frac{n!}{\prod_{i=1}^k x_i}\prod_{i=1}^k p_i^{x^i}


where :math:`\mathbf{p}` is a k-dimensional vector defined as :math:`\mathbf{p} = (p_1, p_2, \ldots, p_k)` with the probability
associated to each possible outcome.



The definition of a random variable following a multinomial distribution is done as follows:

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 93-97



Multivariate-Normal
~~~~~~~~~~~~~~~~~~~~~~~

A multivariate-normal (or Gaussian) defines a set of  normal-distributed variables which are assumed
to be idependent. In other words, the covariance matrix is diagonal.

A single multivariate-normal distribution defined on :math:`\mathbb{R}^2` can be defined as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 140-143


Normal
~~~~~~~~~~~~~~~

The normal (or Gaussian) distribution is a continuous probability distribution with the following density function

.. math::

   p(x;\mu,\sigma) = \frac{1}{2\sigma} exp \left( - \frac{|x - \mu |}{\sigma}\right)

where :math:`\mu`  is the mean or expectation of the distribution, :math:`\sigma`  is the standard deviation, and :math:`\sigma^{2}` is the variance.


A normal distribution can be defined as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 105-109

where the input parameter ``loc`` and ``scale`` correspond to :math:`\mu` and :math:`\sigma` respectively.




Poisson
~~~~~~~~~~~~~~~

The Poisson distribution is a discrete probability distribution for modeling the number of times an event occurs
in an interval of time or space. Its probability mass function is


.. math::

   p(x;\lambda) = e^{- \lambda} \frac{\lambda^x}{x!}

where :math:`\lambda` is the rate or number of events per interval.

A Poisson distribution can be defined as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 116-120







Uniform
~~~~~~~~~~~~~~~

The continuous uniform distribution or rectangular distribution assings the same probability to any :math:`x`  in
the interval :math:`[a,b]`.


.. math::

   p(x;a,b) =\left\{\begin{array}{cc} \frac{1}{b-a} & \mathrm{if\ } x\in [a,b]\\
    0 & \mathrm{if\ } x\not\in [a,b] \\ \end{array} \right.


A uniform distribution can be defined as follows.

.. literalinclude:: ../../examples/supported_distributions.py
   :language: python
   :lines: 128-132


where the input parameters ``low`` and ``high`` correspond to the lower and upper bounds of the interval :math:`[a,b]`.



