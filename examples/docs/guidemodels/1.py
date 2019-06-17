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
s = m.sample()



# compute the log probability
m.log_prob(s)


m.sample(size=5)

"""
>>> m.sample(size=5)
{'w': array([[ 1.4588978 , -0.78119284]], dtype=float32), 
 'z': array([[-0.30752778],
       [-0.17902303],
       [ 0.37584424],
       [ 2.1763606 ],
       [ 1.4520675 ]], dtype=float32), 
 'x': array([[-0.7112326 ,  1.3398352 ],
       [ 0.2738361 ,  0.0637802 ],
       [-1.7577622 , -0.26795918],
       [ 3.2689626 , -2.3581226 ],
       [ 2.539008  , -0.13005853]], dtype=float32)}
"""