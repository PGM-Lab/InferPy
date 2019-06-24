import inferpy as inf
import tensorflow as tf
import numpy as np


@inf.probmodel
def simple(mu=0):
    # global variables
    theta = inf.Normal(mu, 0.1, name="theta")

    # local variables
    with inf.datamodel():
        x = inf.Normal(theta, 1, name="x")

m = simple()

#17

"""
>>> m = simple()
>>> type(m)
<class 'inferpy.models.prob_model.ProbModel'>
"""

# 25

"""
>>> m.prior().sample()
{'theta': -0.074800275, 'x': array([0.07758344], dtype=float32)}

>>> m.prior().parameters()
{'theta': {'name': 'theta',
  'allow_nan_stats': True,
  'validate_args': False,
  'scale': 0.1,
  'loc': 0},
 'x': {'name': 'x',
  'allow_nan_stats': True,
  'validate_args': False,
  'scale': 1,
  'loc': 0.116854645}}
"""



"""
>>> m.vars["theta"]
<inf.RandomVariable (Normal distribution) named theta/, shape=(), dtype=float32>
"""





#55

"""
>>> m2 = simple(mu=5)
>>> m==m2
False
"""




#66
"""
>>> sess = inf.get_session()
>>> sess.run(m2.vars["x"].loc)
4.849595
"""

# 73

"""
>>> inf.models.random_variable.distributions_all
['Autoregressive', 'BatchReshape', 'Bernoulli', 'Beta', 'BetaWithSoftplusConcentration',
 'Binomial', 'Categorical', 'Cauchy', 'Chi2', 'Chi2WithAbsDf', 'ConditionalTransformedDistribution',
  'Deterministic', 'Dirichlet', 'DirichletMultinomial', 'ExpRelaxedOneHotCategorical', '
  Exponential', 'ExponentialWithSoftplusRate', 'Gamma', 'GammaGamma', 
  'GammaWithSoftplusConcentrationRate', 'Geometric', 'GaussianProcess', 
  'GaussianProcessRegressionModel', 'Gumbel', 'HalfCauchy', 'HalfNormal', 
  'HiddenMarkovModel', 'Horseshoe', 'Independent', 'InverseGamma',
   'InverseGammaWithSoftplusConcentrationRate', 'InverseGaussian', 'Kumaraswamy',
   'LinearGaussianStateSpaceModel', 'Laplace', 'LaplaceWithSoftplusScale', 'LKJ',
  'Logistic', 'LogNormal', 'Mixture', 'MixtureSameFamily', 'Multinomial',
   'MultivariateNormalDiag', 'MultivariateNormalFullCovariance', 'MultivariateNormalLinearOperator',
   'MultivariateNormalTriL', 'MultivariateNormalDiagPlusLowRank', 'MultivariateNormalDiagWithSoftplusScale',
   'MultivariateStudentTLinearOperator', 'NegativeBinomial', 'Normal', 'NormalWithSoftplusScale', 
   'OneHotCategorical', 'Pareto', 'Poisson', 'PoissonLogNormalQuadratureCompound', 'QuantizedDistribution',
   'RelaxedBernoulli', 'RelaxedOneHotCategorical', 'SinhArcsinh', 'StudentT', 'StudentTWithAbsDfSoftplusScale', 
   'StudentTProcess', 'TransformedDistribution', 'Triangular', 'TruncatedNormal', 'Uniform', 'VectorDeterministic',
   'VectorDiffeomixture', 'VectorExponentialDiag', 'VectorLaplaceDiag', 'VectorSinhArcsinhDiag', 'VonMises', 
   'VonMisesFisher', 'Wishart', 'Zipf']
"""
