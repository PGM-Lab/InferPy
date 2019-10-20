import inferpy as inf
import numpy as np

# definition of a generic model
@inf.probmodel
def pca(k,d):
    w = inf.Normal(loc=np.zeros([k,d]), scale=1, name="w")      # shape = [k,d]
    with inf.datamodel():
        z = inf.Normal(np.ones(k),1, name="z")                # shape = [N,k]
        x = inf.Normal(z @ w , 1, name="x")                     # shape = [N,d]


# create an instance of the model
m = pca(k=1,d=2)

# create another instance of the model
m2 = pca(k=3, d=10)


# get a sample from the pior distirbution
s = m.prior().sample()



# compute the log probability
m.prior(data=s).log_prob()


m.prior().sample(size=5)

