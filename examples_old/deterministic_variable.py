import inferpy as inf
import edward as ed
import tensorflow as tf


# empty initialized variable
x = inf.models.Deterministic()
x.dist = ed.models.Normal(0.,1.)


# deterministic variable with returning a tensor

x = inf.models.Deterministic()
x.base_object = tf.constant(1.)


# deterministic variable returning an InferPy variable

a = inf.models.Normal(0,1)
b = inf.models.Normal(2,2)

x = a + b

type(x)  # <class 'inferpy.models.deterministic.Deterministic'>

