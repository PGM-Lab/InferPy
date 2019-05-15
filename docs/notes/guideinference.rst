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
   :language: python
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
   :language: python
   :lines: 31-40




In the 'Q' model we should include a q distribution for every non observed variable in 
the 'P' model. These varaiables are also objects of class ```inferpy.RandomVariable```.
However, their parameters might be of type ```inf.Parameter```, which are objects
encapsulating TensorFlow trainable variables.


Finally, when defining the inference algorithm, we must specify an instance
of the 'Q' model:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 44-45


Then we must instantiate our 'P' model and fit the data with the inference
algorithm defined.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 46-49


The output generated will be similar to:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :lines: 52-56



Finally we can access to the dictionary with the posterior distributions:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 64-65

Custom Loss function
---------------------------------

Following InferPy guiding principles, users can further configure the inference algorithm.
For example, we might be interested in defining our own function to minimise. As
an example, we define the following function taking as input parameters the instances
of the P and Q models, and the dictionary with the observations. Note that the output of this function must be a tensor.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 75-100



For using our own loss function, we simply have to pass this function to the
input parameter ``loss`` in the inference method constructor. For example:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 105-108


After this, the rest of the code remains unchanged.


Coding the Optimization Loop
-------------------------------------

As an InferPy model encapsulates an equivalent one in Edward, we can extract the
required tensors and explicitly code the optimization loop. However, this
is **not** recommended for non-expert users in TensorFlow.


First, we get the tensor for the ELBO, but we must first invoke the method
``inf.util.runtime.set_tf_run(False)`` which avoids the evaluation of such tensor.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 121-124


Then we must initialize the optimizer and the session:

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 126-133



Afterwards, we code the loop itself, where the tensor ``train`` must be evaluated
at each iteration for performing each optimization step.

.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 139-142

After the optimization, we can extract the posterior distributions:


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 145-146