.. _proobzoo:


Probabilistic Model Zoo
===========================

In this section, we present the code for implementing some models in Inferpy.


Bayesian Linear Regression
-----------------------------------

Graphically, a (Bayesian) linear regression can be defined as follows,

.. figure:: ../_static/img/linear_regression.png
   :alt: Bayesian Linear Regression
   :scale: 60 %
   :align: center

   Bayesian Linear Regression

The InferPy code for this model is shown below,


.. literalinclude:: ../../examples/probzoo/linear_regression.py
   :language: python

--------------


Bayesian Logistic Regression
------------------------------------

Graphically, a (Bayesian) logistic regression can be defined as follows,

.. figure:: ../_static/img/logistic_regression.png
   :alt: Bayesian Logistic Regression
   :scale: 60 %
   :align: center

   Bayesian Linear Regression

The InferPy code for this model is shown below,



.. literalinclude:: ../../examples/probzoo/log_regression.py
   :language: python


--------------


Linear Factor Model (PCA)
-------------------------

A linear factor model allows to perform principal component analysis (PCA). Graphically,
 it can be defined as follows,

.. figure:: ../_static/img/pca.png
   :alt: Linear Factor Model (PCA)
   :scale: 100 %
   :align: center

   Linear Factor Model (PCA)

The InferPy code for this model is shown below,

.. literalinclude:: ../../examples/probzoo/pca.py
   :language: python
   :lines: 8-32


--------------

Non-linear Factor Model (NLPCA)
--------------------------------------


Similarly to the previous model, the Non-linear PCA can be graphically defined as follows,

.. figure:: ../_static/img/nlpca.png
   :alt: Non-linear PCA
   :scale: 100 %
   :align: center

   Non-linear PCA

Its code in InferPy is shown below,

.. literalinclude:: ../../examples/probzoo/nlpca.py
   :language: python
   :lines: 8-68





