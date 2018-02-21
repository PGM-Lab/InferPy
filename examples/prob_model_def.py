import inferpy as inf
from inferpy.models import Normal


x = Normal(loc=1., scale=1.)
y = Normal(loc=x, scale=1.)

x.sample(3)


p = inf.ProbModel(varlist=[x,y])

p.sample(10)

y.shape

inf.ProbModel.is_active()






v=1


p = inf.ProbModel()
p.add_var(x)


p.varlist
# TODO:
# compile method

import inferpy as inf
from inferpy.models import *


with inf.ProbModel() as prb:
    x = Normal(loc=5., scale=1.)


import tensorflow as tf

v = [1,2]
x.dist.prob(tf.cast(v, tf.float64))