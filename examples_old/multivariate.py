import inferpy as inf
import tensorflow.contrib.distributions as tfd
import edward as ed
import tensorflow as tf

sess = inf.get_session()

with inf.replicate(size=10):
    x = inf.models.MultivariateNormalDiag(
        loc=[1., -1],
        scale_diag=[1, 2.], dim=3
    )


x.shape         #  [10, 3, 2]
x.batches       #  10
x.dim           #  3
x.event_shape   # [2]

