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

Any random variable in InferPy encapsulates an equivalent one in Edward 2, and hence it also has associated
a distribution object from TensorFlow Probability. These can be accessed using the properties ``var`` and
``distribution`` respectively:

.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 12-19


Even more, InferPy random variables inherit all the properties and methods from Edward2 variables or TensorFlow
Probability distributions (in this order or priority). For example:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 26-33


In the previous code, ``value`` is inherited form the encapsulated Edward2 object while ``sample()`` and the
parameter ``loc`` are obtained from the distribution object. Note that the method ``sample()`` returns
an evaluated tensors. In case of desiring it not to be evaluated, simply use the input parameter ``tf_run`` as follows.

.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 38-39



Following Edward's approach, we (conceptually) partition a random variable's shape into three groups:

- *Batch shape* describes independent, not identically distributed draws. Namely, we may have a set of (different) parameterizations to the same distribution.
- *Sample shape* describes independent, identically distributed draws from the distribution.
- *Event shape* describes the shape of a single draw (event space) from the distribution; it may be dependent across dimensions.



The previous attributes can be accessed by ``x.batch_shape``, ``x.sample_shape`` and ``x.event_shape``,
respectively.  When declaring random variables, the *batch_shape* is obtained from the distribution
parameters. For as long as possible, the parameters will be broadcasted. With this in mind, all the definitions in the
following code are equivalent.

.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 53-57



The ``sample_shape`` can be explicitly stated using the input parameter
sample_shape, but this only can be done outside a model definition.
Inside of ``inf.probmodels``, the sample_shape is fixed by ``with inf.datamodel(size = N)`` (using the size argument
when provided, or in runtime depending on the observed data).



.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 66-69


Finally, the *event shape* will only be consider in some distributions. This is the case of the
multivariate Gaussian:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 75


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 83-90




Note that indexing over all the defined dimenensions is supported:


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 103-114


Moreover, we may use indexation for defining new variables whose indexes may be other (discrete) variables.


.. literalinclude:: ../../examples/docs/guidemodels/2.py
   :language: python3
   :lines: 123-125




Probabilistic Models
--------------------
A **probabilistic model** defines a joint distribution over observable
and hidden variables, i.e., :math:`p(\mathbf{w}, \mathbf{z}, \mathbf{x})`. Note that a
variable might be observable or hidden depending on the fitted data. Thus this is
not specified when defining the model.


A probabilistic model is defined by decorating any function with ``@inf.probmodel``.
The model is made of any variable defined inside this function. A simple example is shown
below.

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 6-13


Note that any variable in a model can be initialized with a name. If not provided, names generated
automatically will be used. However, it is highly convenient to explicitly specify the name of a random variable because
in this way it will be able to be referenced in some inference stages.


The model must be **instantiated** before it can be used. This is done by simple
invoking the function (which will return a probmodel object).

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 20-22


Now we can use the model with the prior probabilities. For example,
we might get a sample or access to the distribution parameters:

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 28-41

or to extract the variables:


.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 47-48

We can create new and different instances of our model:

.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 58-60





Supported Probability Distributions
-----------------------------------


Supported probability distributions are located in the package ``inferpy.models``. All of them
have ``inferpy.models.RandomVariable`` as superclass. A list with all the supported distributions can be obtained as
as follows.


.. literalinclude:: ../../examples/docs/guidemodels/3.py
   :language: python3
   :lines: 76-94

Note that these are all the distributions in Edward 2 and hence in TensorFlow Probability. Their
input parameters will be the same.
