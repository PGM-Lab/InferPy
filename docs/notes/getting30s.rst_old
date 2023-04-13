Getting Started
================

Installation
-----------------

Install InferPy from PyPI:

.. code:: bash

   $ python -m pip install inferpy


For further details, check the `Installation <installation.html>`_ section.



30 seconds to InferPy
--------------------------

The core data structure of InferPy is a **probabilistic model**,
defined as a set of **random variables** with a conditional independence
structure. A **random variable** is an object
parameterized by a set of tensors.

Let's look at a simple non-linear **probabilistic component analysis** model (NLPCA). Graphically the model can
be defined as follows,

.. figure:: ../_static/img/nlpca.png
   :alt: Non-linear PCA
   :scale: 60 %
   :align: center

   Non-linear PCA

We start by importing the required packages and defining the constant parameters in the model.

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 1-11

A model can be defined by decorating any function with ``@inf.probmodel``. The model is fully specified by
the variables defined inside this function:

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 14-22


The construct ``with inf.datamodel()``, which resembles the **plateau notation**, will replicate
N times the variables enclosed, where N is the data size.


In the previous model, the input argument ``decoder`` must be a function implementing a neural network.
This can be defined outside the model as follows.


.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 27-30


Now, we can instantiate our model and obtain samples (from the prior distributions).

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 48-52




In variational inference, we need to define a Q-model as follows.


.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 36-42



Afterwards, we define the parameters of the inference algorithm and fit the model to the data.



.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 62-66


The inference method can be further configured. But, as in Keras, a core
principle is to try to make things reasonably simple, while allowing the
user the full control if needed.



Finally, we might extract the posterior of ``z``, which is basically the hidden representation
of the data.



.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 71-73
