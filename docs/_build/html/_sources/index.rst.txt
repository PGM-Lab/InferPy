.. InferPy documentation master file, created by
   sphinx-quickstart on Fri Nov  3 12:26:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

InferPy: Deep Probabilistic Modelling Made Easy
===============================================

.. image:: _static/img/logo.png
   	:scale: 90 %
   	:align: center

|


InferPy is a high-level API for probabilistic modeling written in Python and 
capable of running on top of Edward and Tensorflow. InferPy's API is 
strongly inspired by Keras and it has a focus on enabling flexible data processing, 
easy-to-code probablistic modelling, scalable inference and robust model validation. 

Use InferPy is you need a probabilistic programming language that:

* Allows easy and fast prototyping of hierarchical probabilistic models with a simple and user friendly API inspired by Keras. 
* Defines probabilistic models with complex probabilistic constructs containing deep neural networks.   
* Automatically creates computational efficient batched models without the need to deal with complex tensor operations.
* Run seamlessly on CPU and GPU by relying on Tensorflow. 

.. * Process seamlessly small data sets stored on a Panda's data-frame, or large distributed data sets by relying on Apache Spark.

InferPy is to Edward what Keras is to Tensorflow
-------------------------------------------------
InferPy's aim is to be to Edward what Keras is to Tensorflow. Edward is a general purpose
probabilistic programing language, like Tensorflow is a general computational engine. 
But this generality comes a at price. Edward's API is
verbose and is based on distributions over Tensor objects, which are n-dimensional arrays with 
complex semantics operations. Probability distributions over Tensors are powerful abstractions 
but it is not easy to operate with them. InferPy's API is no so general like Edward's API 
but still covers a wide range of powerful and widely used probabilistic models, which can contain
complex probability constructs containing deep neural networks.  


.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Quick Start
   
   notes/getting30s
   notes/gettingGuiding
   notes/gettingInstallation


.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Guides
   
   notes/guidemodels
   notes/guideinference
   notes/guidebayesian
   notes/guidevalidation
   notes/guidedata

.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Model Zoo
   
   notes/probzoo



.. Indices and tables
  ==================

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`


.. role:: bash(code)
   :language: bash
.. role:: python(code)
   :language: python
