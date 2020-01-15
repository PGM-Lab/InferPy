1.3.0
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:

- Integration with Bayesian Layers from TFP.
- Keras models can be defined inside InferPy models.
- Inference with MCMC.
- Documentation update.
- Fixed bugs #200, #201, #202.



**Release Date**: 16/01/2020
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)


1.2.3
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:

- Bug detected at #195: false dependency is created between RVs which
are acenstors of a trainable layer.
- Documentation updated.



**Release Date**: 18/10/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




1.2.2
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:

- Hotfix at #193, dependency changed of ``tensorflow-probability`` from ``>=0.5.0,<0.1.0``
 to ``>=0.5.0,<0.8.0``.



**Release Date**: 10/10/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)



1.2.1
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:

- Function ``inf.MixtureGaussian`` encapsulating ``ed.MixtureSameFamily``.
- Documentation updated.



**Release Date**: 19/09/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)



1.2.0
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- Data handling from memory and CSV files.
- Renamed inferpy.datasets to inferpy.data.
- Internal code enhancements.
- Documentation extended.
- Fixed some bugs.


**Release Date**: 29/08/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




1.1.3
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- Fixed some bugs related to posterior predictive computation.
- Small internal enhancement.


**Release Date**: 26/08/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)


1.1.1
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- Updated requirements.
- New extra requirements: visualization, datasets.


**Release Date**: 08/08/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)





1.1.0
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- API for prior, posterior, and posterior_predictive queries.
- GPU support.
- Small changes in code structure.
- Fixed compatibility issue with TFP 0.7.0.
- Documentation updated.
- Fixed some bugs.


**Release Date**: 04/07/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)





1.0.0
=======
InferPy is a high-level API for defining probabilistic models containing deep neural networks in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- Extensive re-design of the API.
- Compatible with TFP/Edward 2.
- Edward 1 is not further supported.


**Release Date**: 27/05/2019
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)



0.2.1
========
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.


Changes:
- batch parameter in random variable definitions.
- Changes in documentation.
- Name reference to replicate constructs.
- Predefiend and custom parametrised models (inf.models.predefiend)
- Version flag moved to inferpy/\_\_init\_\_.py
- Fixed some bugs.

**Release Date**: 23/11/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)






0.2.0
========
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
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
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
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
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
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
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
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
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
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
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

Changes:
- Fixed some bugs
- RandomVariable base class
- Optional parameter for returning a TensorFlow object
- Latent variables
- Dependency between variables
- Definition of probabilistic models
- Inference with KLqp

**Release Date**: 02/03/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)




0.0.1
============
InferPy is a high-level API for probabilistic modeling written in Python and capable of running on top of
Edward and TensorFlow. InferPy’s API is strongly inspired by Keras and it has a focus on enabling flexible
data processing, easy-to-code probabilistic modeling, scalable inference and robust model validation.

This version includes the basic functionality:

- Normal distributions
- Replicate construct

**Release Date**: 09/02/2018
**Further Information**: [Documentation](http://inferpy.readthedocs.io/)