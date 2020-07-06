Comparison: Logistic Regression
===================================

Here, the InferPy code is compared with other similar frameworks.
A logistic regression will be considered.

Setting up
-------------

First the required packages are imported. Variable ``d`` is the number of predictive
attributes while ``N`` is the number of observations.

.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/log_regression.py
         :language: python3
         :lines: 3-8

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_logregression.py
         :language: python3
         :lines: 3-7

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_logregression.py
         :language: python3
         :lines: 3-11






Model definition
-------------------------

Models are defined as functions. In case of InferPy these must be decoraed
with ``@inf.probmodel``. Inspired in Pyro, InferPy uses construct ``inf.datamodel``
for simplifying the definition of the variables dimension. In the following
code fragments, P and Q models are defined.



.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/log_regression.py
         :language: python3
         :lines: 14-32

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_logregression.py
         :language: python3
         :lines: 16-34

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_logregression.py
         :language: python3
         :lines: 22-34


Sample form the pior model
--------------------------------

Now we can sample from the P-model in which the global parameters are
fixed. As it can be observed below, this is more complex in TFP.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/log_regression.py
         :language: python3
         :lines: 39-45

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_logregression.py
         :language: python3
         :lines: 47-66

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_logregression.py
         :language: python3
         :lines: 41-42




Inference
--------------------------

Using the data generated, variational inference can be done as follows.
This is quite simple with our package, while TFP and Pyro require the user to implement
optimization loop.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/log_regression.py
         :language: python3
         :lines: 51-52

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_logregression.py
         :language: python3
         :lines: 72-110

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_logregression.py
         :language: python3
         :lines: 46-55


Usage of the inferred model
-----------------------------

Finally, the posterior distributions of the global parameters ``w``
can be shown ``w0``. From the posterior predictive distribution,
samples can be generated as follows.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/log_regression.py
         :language: python3
         :lines: 58-69

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_logregression.py
         :language: python3
         :lines: 115-125

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_logregression.py
         :language: python3
         :lines: 61-71

