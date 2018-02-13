import inferpy as inf
from inferpy.models import *


x = Normal(loc=0., scale=1.)
y = Normal(loc=1., scale=1.)

dist_list = [x,y]

p = inf.ProbModel(dist_list)

inf.ProbModel.is_active()






v=1


p = inf.ProbModel()
p.add_dist(x)


p.distlist
# TODO:
# check append
# object string name
# compile method

import inferpy as inf
from inferpy.models import *


with inf.ProbModel() as prb:
    x = Normal(loc=5., scale=1.)


prb.distlist

for d in dist_list:
    if isinstance(d, RandomVariable)==False:
        raise ValueError("The input argument is not a list of RandomVariables")


print(inf.ProbModel.is_active())

with inf.ProbModel(dist_list) as prb:
    print(inf.ProbModel.is_active())
    with inf.ProbModel(dist_list) as prb:
        print("")


