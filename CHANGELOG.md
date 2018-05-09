0.1.0
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probablistic modelling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- Qmodel class
- New distributions: Gamma, Bernoulli, InverseGamma, Laplace
- inferpy.models.ALLOWED_VARS is a list with all the types of variables (i.e., distributions) allowed.
- infMethod argument in compile method
- inferpy.case function wrapping tensorflow.case
- Boolean operators
- Correlated samples from ProbModel




**Release Date**: 09/05/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.0.3
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probablistic modelling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- New distributions: Beta, Exponential, Uniform, Poisson, Categorical, Multinomial, Dirichlet
- Integration with pandas

**Release Date**: 25/03/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)





0.0.2
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probablistic modelling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- RandomVariable base class
- Optional parameter for returning a Tensorflow object
- Latent variables
- Dependency between variables
- Definition of probabilistic models
- Inference with KLqp

**Release Date**: 02/03/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.0.1
============
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probablistic modelling, scalable inference and robust model validation.

This version includes the basic functionality:

- Normal distributions
- Replicate construct

**Release Date**: 09/02/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)