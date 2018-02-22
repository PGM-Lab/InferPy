import inferpy as inf
from inferpy.models import Normal


x = Normal(loc=1., scale=1.)
y = Normal(loc=x, scale=1.)



p = inf.ProbModel(varlist=[x,y])

p.sample(10)


