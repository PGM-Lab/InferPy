InferPy: Deep Probabilistic Modeling Made Easy
==============================================

InferPy is a high-level API for probabilistic modeling written in Python
and capable of running on top of Edward and Tensorflow. InferPy\'s API
is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference
and robust model validation.

Use InferPy is you need a probabilistic programming language that:

-   Allows easy and fast prototyping of hierarchical probabilistic
    models with a simple and user friendly API inspired by Keras.
-   Defines probabilistic models with complex probabilistic constructs
    containing deep neural networks.
-   Automatically creates computational efficient batched models without
    the need to deal with complex tensor operations.
-   Run seamlessly on CPU and GPU by relying on Tensorflow.

InferPy is to Edward what Keras is to Tensorflow
------------------------------------------------

InferPy\'s aim is to be to Edward what Keras is to Tensorflow. Edward is
a general purpose probabilistic programing language, like Tensorflow is
a general computational engine. But this generality comes a at price.
Edward\'s API is verbose and is based on distributions over Tensor
objects, which are n-dimensional arrays with complex semantics
operations. Probability distributions over Tensors are powerful
abstractions but it is not easy to operate with them. InferPy\'s API is
no so general like Edward\'s API but still covers a wide range of
powerful and widely used probabilistic models, which can contain complex
probability constructs containing deep neural networks.

For more details about InferPy, check the online [documentation
documentation](https://inferpy.readthedocs.io).