Guide to Approximate Inference
==============================

Variational Inference
------------------------------------------

The API defines the set of algorithms and methods used to perform
inference in a probabilistic model :math:`p(x,z,\theta)` (where
:math:`x` are the observations, :math:`z` the local hidden variables,
and :math:`\theta` the global parameters of the model). More precisely,
the inference problem reduces to compute the posterior probability over
the latent variables given a data sample
:math:`p(z,\theta | x_{train})`, because by looking at these
posteriors we can uncover the hidden structure in the data. Let us consider the
following model:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 6-11




In this model, the posterior over the local hidden variables :math:`p(w_n|x_{train})`
tell us the latent vector representation of the sample :math:`x_n`, while the posterior
over the global variables :math:`p(\mu|x_{train})` tells us which is the affine transformation
between the latent space and the observable space. 

.. where the centroids of the data are, while :math:`p(z_n|x_{train})` shows us to which centroid every data point belongs to.

InferPy inherits Edward's approach an consider approximate inference
solutions,

.. math::  q(z,\theta) \approx p(z,\theta | x_{train})



in which the task is to approximate the posterior
:math:`p(z,\theta | x_{train})` using a family of distributions,
:math:`q(z,\theta; \lambda)`, indexed by a parameter vector
:math:`\lambda`.



For making the inference, we must define a model 'Q' for approximating the
posterior distribution. This is also done by defining a function decorated
with ``@inf.probmodel``:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 31-40




In the 'Q' model we should include a q distribution for every non-observed variable in
the 'P' model. These varaiables are also objects of class ``inferpy.RandomVariable``.
However, their parameters might be of type ``inf.Parameter``, which are objects
encapsulating TensorFlow trainable variables.


Then, we set the parameters of the inference algorithm. In case of variational inference
(VI) we must specify an instance of the 'Q' model and the number of ``epochs`` (i.e.,
iterations). For example:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 44-45

VI can be further configured by setting the parameter ``optimizer`` which
indicates the TensorFlow optimizer to be used (AdamOptimizer by default).

Stochastic VI is similarly specified but has an additional input parameter for specifying
the batch size:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 144-144


Then we must instantiate our 'P' model and fit the data with the inference
algorithm defined.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 46-49


The output generated will be similar to:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :lines: 52-56



Finally, we can access to the parameters of the posterior distributions:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 62-67




Queries
---------

The queries syntax allows using the probabilistic models specifying a type of knowledge: prior, posterior or posterior
predictive. That means that, for example, we can generate new instances from the prior knowledge (using the initial
model definition), or the posterior/posterior predictive knowledge (once the model has been trained using input data).
There are two well-differentiated parts: the query definition and the action function. The action functions can be applied
on ``Query`` objects to:

- ``sample``: samples new data.

- ``log_prob``: computes the log prob given some evidence (observed variables).

- ``sum_log_prob``: the same as `log_prob`, but computes the sum of the log prob for all the variables in the probabilistic model.

- ``parameters``: returns the parameters of the Random Variables (i.e.: loc and scale for Normal distributions).


Building Query objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a probabilistic model object, i.e.: model, we can build ``Query`` objects by calling the ``prior()``,
``posterior()`` or ``posterior_predictive()`` methods of the ``probmodel`` class. All these accepts the same two
arguments:

- ``target_names``: A string or list of strings that correspond to random variable names. These random variables
  are the targets of our queries (in other words, the random variables that we want to use when calling an action).

- ``data``: A dict that contains as keys the name of random variables, and the values the observed data for that
  random variables. By default, it is an empty dict.

Let's define each function:


- ``prior()``: This function returns ``Query`` objects that use the random variables initially defined in the model
  when applying the actions. It just uses prior knowledge and can be used once the model object is created.

- ``posterior()``: This function returns ``Query`` objects that use the expanded random variables defined and
  fitted after the training process. It uses the posterior knowledge and can be used only after calling the ``fit``
  function. The target variables allowed are those not observed during the training process.

- ``posterior_predictive()``: This function is similar to the ``posterior``, but he target variables allowed
  in this function are those observed during the training process.


Action functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Action functions allow getting the desired information from the ``Query`` objects. As described before, actually
there are four functions:

- ``sample(size)``: Generates _size_ instances (by default ``size=1``). It returns a dict, where the keys are the random variable names and the values are the sample data. If there is only one target name, then only the sample data is returned.

- ``log_prob()``: computes the log prob given the evidence specified in the ``Query`` object. It returns a dict,
  where the keys are the random variable names and the values are the log probs. If there is only one target name,
  then only the log prob is returned.

- ``sum_log_prob()``: the same as ``log_prob``, but computes the sum of the log prob for all the variables
  in the probabilistic model.

- ``parameters(names)``: returns the parameters of the Random Variables. If ``names`` is ``None`` (by default)
  it returns all the parameters of all the random variables. If ``names`` is a string or a list of strings,
  that corresponds to parameter names, then it returns the parameters of the random variables that match with any name provided in the _names_ argument. It returns a dict, where the keys are the random variable names and
  the values are the dict of parameters (name of parameter: parameter value). If there is only one target name,
  then only the dict of parameters for such a random variable is returned.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a full example of the usage of the queries, we might consider the following code.

.. literalinclude:: ../../examples/docs/guideinference/2.py
   :language: python3



Custom Loss function
---------------------------------

Following InferPy guiding principles, users can further configure the inference algorithm.
For example, we might be interested in defining our own function to minimize. As
an example, we define the following function taking as input parameters the random variables
of the P and Q models (we assume that their sample sizes are consistent with the plates in the model). Note that the output of
this function must be a tensor.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 77-89



For using our defined loss function, we simply have to pass this function to the
input parameter ``loss`` in the inference method constructor. For example:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 93-97


After this, the rest of the code remains unchanged.



