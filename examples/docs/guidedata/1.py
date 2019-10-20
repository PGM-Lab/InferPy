from inferpy.data.loaders import CsvLoader, SampleDictLoader, DataLoader
import numpy as np

# CSV files

from inferpy.data.loaders import CsvLoader

data_loader = CsvLoader(path="./tests/files/dataxy_0.csv")

file_list = [f"./tests/files/dataxy_{i}.csv" for i in [0,1]]
data_loader = CsvLoader(path=file_list)


# data in memory

from inferpy.data.loaders import SampleDictLoader

samples = {"x": np.random.rand(1000), "y": np.random.rand(1000)}
data_loader = SampleDictLoader(sample_dict=samples)



# Properties

data_loader.size

data_loader.variables

# data_loader.has_header


"""
>>> data_loader.size
1000
>>> data_loader.variables
['x', 'y']
>>> data_loader.has_header
True
"""


# Mapping

data_loader = CsvLoader(path="./tests/files/dataxy_0.csv", var_dict={"x1":[0], "x2":[1]})

"""
>>> data_loader.variables
['x1', 'y2']
"""

data_loader = CsvLoader(path="./tests/files/dataxy_0.csv", var_dict={"A":[0,1]})

"""
>>> data_loader.variables
['A']
"""


# Extracting data

data_loader.to_dict()

data_loader.to_tfdataset(batch_size=50)

"""
>>> data_loader.to_dict() 
{'x': array([1.54217069e-02, 3.74321848e-02, 1.29080105e-01, ... ,8.44103262e-01]),
 'y': array([1.49197044e-01, 4.19856938e-01, 2.63596605e-01, ... ,1.20826740e-01])}

>>> data_loader.to_tfdataset(batch_size=50)
<DatasetV1Adapter shapes: OrderedDict([(x, (50,)), (y, (50,))]), 
types: OrderedDict([(x, tf.float32), (y, tf.float32)])>

"""


# Use with InferPy models


import inferpy as inf
import tensorflow as tf

@inf.probmodel
def linear_reg(d):
    w0 = inf.Normal(0, 1, name="w0")
    w = inf.Normal(tf.zeros([d,1]), 1, name="w")

    with inf.datamodel():
        x = inf.Normal(tf.ones([d]), 2, name="x")
        y = inf.Normal(w0 + x @ w, 1.0, name="y")


@inf.probmodel
def qmodel(d):
    qw0_loc = inf.Parameter(0., name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

    qw_loc = inf.Parameter(tf.zeros([d,1]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d,1]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")


# create an instance of the model
m = linear_reg(d=1)
vi = inf.inference.VI(qmodel(1), epochs=100)


m.fit(data={"x": np.random.rand(1000,1), "y": np.random.rand(1000,1)}, inference_method=vi)



data_loader = CsvLoader(path="./tests/files/dataxy_with_header.csv")
m.fit(data=data_loader, inference_method=vi)



data_loader = CsvLoader(path="./tests/files/dataxy_no_header.csv", var_dict={"x":[0], "y":[1]})
m.fit(data=data_loader, inference_method=vi)

