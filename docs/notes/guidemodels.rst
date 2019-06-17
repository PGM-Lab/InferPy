Guide to Probabilistic Models
======================================

Getting Started with Probabilistic Models
------------------------------------------

InferPy focuses on *hierarchical probabilistic models* structured
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
   :language: python3
   :lines: 4-14



The ``with inf.datamodel()`` sintaxis is used to replicate the
random variables contained within this construct. It follows from the
so-called *plateau notation* to define the data generation part of a
probabilistic model. Every replicated variable is **conditionally
idependent** given the previous random variables (if any) defined
outside the **with** statement. The plateau size will be later automatically calculated,
so there is not need to specify it. Yet, this construct has an optional input parameter for specifying
its size, e.g., ``with inf.datamodel(size=N)``. This should be consistent with the size of
our data.

.. Internally, ``with inf.replicate(size = N)`` construct modifies the
   random variable shape by adding an extra dimension. For the above
   example, z\_n's shape is [N,1], and x\_n's shape is [N,d].


Random Variables
----------------

Any random variable in InferPy encapsulates an equivalent one in Edward (i.e., version 2), and hence it also has associated
a distribution object from TensorFlow Probability. These can be access using the properties ``var`` and
``dist`` respectively:

.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 12-19


Even more, InferPy random variables inherit all the properties and methods from Edward variables. For
example:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 25-29



Following Edward's approach, we (conceptually) partition a random variable's shape into three groups:

- *Batch shape* describes independent, not identically distributed draws. Namely, we may have a set of (different) parameterizations to the same distribution.
- *Sample shape* describes independent, identically distributed draws from the distribution.
- *Event shape* describes the shape of a single draw (event space) from the distribution; it may be dependent across dimensions.


When declaring random variables, InferPy provides different ways for defining previous shapes. First,
the batch shape could be obtained from the distribution parameter shapes or explicitly stated using the input parameter
``batch_shape``.  With this in mind, all the definitions in the following code are equivalent.


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 38-44


The ``with inf.datamodel(size = N)`` sintaxis is used to specify the sample shape. Alternatively,
we might explicitly state it using the input paramenter ``sample_shape``. This is actually inherit
from Edward.

.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 50-53


Finally, the sample shape will only be consider in some distributions. This is the case of the
multivariate Gaussian:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 59


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 67-74




Note that indexing is supported:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 85-95


Moreover, we may use indexation for defining new variables whose indexes may be other (discrete) variables.



Probabilistic Models
--------------------
A **probabilistic model** defines a joint distribution over observable
and hidden variables, :math:`p(\mathbf{w}, \mathbf{z}, \mathbf{x})`. Note that a
variable might be observable or hidden depending on the fitted data. Thus this is
not specified when defining the model.


A probabilistic model is defined by decorating any function with ``@inf.probmodel``.
The model is made of any variable defined inside this function. A simple example is shown
below.

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 6-13


Note that any variable in a model must be initialized with a name (this
is not required when defining random variables outside the probmodel scope).


The model must be **instantiated** before it can be used. This is done by simple
invoking the function (which will return a probmodel object).

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 20-22


Now we can use the model with the prior probabilities. For example,
we might get a sample:

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 28-29

or extract the variables:


.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 33-34

We can create new and different instances of our model:

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 39-41





Supported Probability Distributions
-----------------------------------


Supported probability distributions are located in the package ``inferpy.models``. All of them
have ``inferpy.models.RandomVariable`` as superclass. A list with all the supported distributions can be obtained as
as follows.


.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 53-71

Note that these are all the distributions in Edward 2 and hence in TensorFlow Probability. Their
input parameters will be the same.
