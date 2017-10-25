INFERPY: A Python Library for Probabilistic Modelling
=====================================================

INFERPY is a high-level API for probabilistic modelling written in Python and capable of running on top of Edward, Tensorflow and Apache Spark. INFERPY's API is strongly inspired by Keras and it has a focus on enabling flexible data processing, simple probablistic modelling, scalable inference and robust model validation. 

Use INFERPY is you need a probabilistic programming language that:

 * Has a simple and user friendly API (inspired by Keras).
 * Allows for easy and fast prototyping of simple probabilistic models or complex probabilistics constructs containing deep neural networks (by relying on Edward).   
 * Run seamlessly on CPU and GPU (by relying on Tensorflow). 
 * Process seamlessly small data sets or large distributed data sets (by relying on Apache Spark). . 

--------


Getting Started: 30 seconds to INFERPY 
--------------------------------------

The core data structures of INFERPY is a a **probabilistic model**, defined as a set of **random variables** with a conditional independence structure. Like in Edward, a **random varible** is an object parameterized by a set of tensors. 

Let's look at a simple examle. We start defining hhe **prior** over the parameters of a **mixture of Gaussians** model: 


.. highlight:: python
   import numpy as np
   import inferpy as inf
   from inferpy.models import Normal, InverseGamma, Dirichlet
