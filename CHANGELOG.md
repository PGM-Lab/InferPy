0.2.0
========
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- Fixed some bugs.
- matmul and dot operations support new input types (numpy, tensors, lists and InferPy variables).
- Extended documentation.
- Moved Qmodel module to inferences package.
- Multidimensional InferPy variables are now indexed in the same way than
numpy arrays (get_item operator).
- Auto-install dependencies fixed.


**Release Date**: 02/10/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)





0.1.2
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

Changes:

- MetropolisHastings (MCMC) inference method
- Creation of empirical q variables
- dot operator
- indexation operator
- MultivariateNormalDiag distribution
- methods mean(), variance() and sddev() for random variables


**Release Date**: 3/08/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.1.1
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- Prediction and evaluation functionality
- Function inf.case_states allows lists of variables as input
- Simple output string for distributions
- Added inf.gather operation
- Transpose is allowed when using inf.matmul
- inf.case works inside a replicate construct
- ProbModel.copy() 
- Code reorganization



**Release Date**: 21/06/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.1.0
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- Qmodel class
- New distributions: Gamma, Bernoulli, InverseGamma, Laplace
- inferpy.models.ALLOWED_VARS is a list with all the types of variables (i.e., distributions) allowed.
- infMethod argument in compile method
- inferpy.case function wrapping tensorflow.case
- Boolean operators
- Correlated samples from ProbModel




**Release Date**: 14/05/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.0.3
=====
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and Tensorflow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

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
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

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
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

This version includes the basic functionality:

- Normal distributions
- Replicate construct

**Release Date**: 09/02/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)