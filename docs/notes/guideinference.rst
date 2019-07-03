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



For making inference, we must define a model 'Q' for approximating the
posterior distribution. This is also done by defining a function decorated
with ``@inf.probmodel``:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 31-40




In the 'Q' model we should include a q distribution for every non observed variable in 
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



Finally we can access to the parameters of the posterior distributions:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 62-67

Custom Loss function
---------------------------------

Following InferPy guiding principles, users can further configure the inference algorithm.
For example, we might be interested in defining our own function to minimise. As
an example, we define the following function taking as input parameters the random variables
of the P and Q models (we assume that their sample sizes are consistent with the plates in the mdoel). Note that the output of
this function must be a tensor.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 77-89



For using our own loss function, we simply have to pass this function to the
input parameter ``loss`` in the inference method constructor. For example:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python3
   :lines: 93-97


After this, the rest of the code remains unchanged.



