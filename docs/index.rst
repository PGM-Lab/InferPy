.. InferPy documentation master file, created by
   sphinx-quickstart on Fri Nov  3 12:26:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.




InferPy: Deep Probabilistic Modeling with TensorFlow Made Easy
=========================================================================


.. image:: _static/img/logo.png
   	:scale: 90 %
   	:align: center

|






InferPy is a high-level API for probabilistic modeling with deep neural networks written in Python and
capable of running on top of TensorFlow. InferPy's API is strongly inspired by Keras and
it has a focus on enabling flexible data processing, easy-to-code probabilistic modeling, scalable inference, and robust model validation.

Use InferPy if you need a probabilistic programming language that:

* Allows easy and fast prototyping of hierarchical probabilistic models with a simple and user-friendly API inspired by Keras.
* Automatically creates computational efficient batch models without the need to deal with complex tensor operations 	and theoretical concepts.
* Run seamlessly on CPU and GPU by relying on TensorFlow, without having to learn how to use TensorFlow.
* Defines probabilistic models with complex probabilistic constructs containing deep neural networks.

.. * Process seamlessly small data sets stored on a Panda's data-frame, or large distributed data sets by relying on Apache Spark.

.. InferPy is to Edward what Keras is to TensorFlow
.. -------------------------------------------------
.. InferPy's aim is to be to Edward what Keras is to TensorFlow. Edward is a general purpose
.. probabilistic programing language, like TensorFlow is a general computational engine.
.. But this generality comes a at price. Edward's API is
.. verbose and is based on distributions over Tensor objects, which are n-dimensional arrays with 
.. complex semantics operations. Probability distributions over Tensors are powerful abstractions 
.. but it is not easy to operate with them. InferPy's API is no so general like Edward's API 
.. but still covers a wide range of powerful and widely used probabilistic models, which can contain
.. complex probability constructs.




Related software
------------------------

Similar frameworks for deep probabilistic modeling are:

- `Edward <http://edwardlib.org>`_
- `TFP/Edward 2 <https://github.com/google/edward2>`_
- `Pyro <https://pyro.ai>`_
- `Stan <https://mc-stan.org>`_
- `Pymc <https://docs.pymc.io>`_

The advantage of InferPy resides on its simplicity. A comparison with respect to TFP/Edward 2 and Pyro is provided
for a `logistic regression <notes/comparative_logreg.html>`_ and for a `variational auto-encoder (VAE) <notes/comparative_vae_mnist.html>`_ using
the MNIST dataset.


Code examples
----------------

A set of examples of models defined with InferPy can be found in the `Probabilistic Model Zoo <notes/probzoo.html>`_ section.
Additionally, more complex examples are provided: `model with Bayesian NN <notes/bayesianNN.html>`_ and a
`mixture density network <notes/mixture.html>`_.



Citation
------------

There are several articles to cite for InferPy. The following one correspond to versions 1.x and describes
the use of InferPy for probabilistic modelling with neural networks. This InferPy version relies on TensorFlow
Probability (TFP) and Edward2.

.. code:: TeX

   @Article{cozar2019inferpy,
        author  = {C{\'o}zar, Javier and Caba{\~n}as, Rafael and Salmer{\'o}n, Antonio and  Masegosa, Andr{\'e}s R},
        title   = {InferPy: Probabilistic Modeling with Deep Neural Networks Made Easy},
        journal = {arXiv preprint arXiv:1908.11161},
        year    = {2019},
   }


On the other hand, the article whose reference is shown below corresponds to the API in verions 0.x which relies
on the first version of Edward, which is no longer under development:


.. code:: TeX

   @article{cabanasInferPy,
        Author = {Caba{\~n}as, Rafael and Salmer{\'o}n, Antonio and Masegosa, Andr{\'e}s R},
        Journal = {Knowledge-Based Systems},
        Publisher = {Elsevier},
        Title = {InferPy: Probabilistic Modeling with TensorFlow Made Easy},
        Year = {2019}
   }


.. .. toctree::
.. :hidden:

.. Home <self>


.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Quick Start
   
   notes/getting30s
   notes/gettingGuiding
   notes/requirements
   notes/installation


.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Inferpy vs Others

   notes/comparative_logreg
   notes/comparative_vae_mnist


.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Guides
   
   notes/guidemodels
   notes/guideinference
   notes/guidebayesian
   notes/guidedata
   notes/advancedsetup
   notes/videotutorials

..   notes/guidevalidation




.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Examples
   
   notes/probzoo
   notes/bayesianNN
   notes/mixture




..  toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/inferpy
   modules/inferpy.contextmanager
   modules/inferpy.data
   modules/inferpy.inference
   modules/inferpy.layers
   modules/inferpy.models
   modules/inferpy.queries
   modules/inferpy.util



.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Other

   notes/contact
   notes/acknowledgements



.. Indices and tables
  ==================

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`
  



.. role:: bash(code)
   :language: bash
.. role:: python(code)
   :language: python3






.. image:: https://travis-ci.org/PGM-Lab/InferPy.svg?branch=master
    :target: https://travis-ci.org/PGM-Lab/InferPy

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0




