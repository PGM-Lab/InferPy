import pytest
from inferpy.data.loaders import CsvLoader, SampleDictLoader, build_data_loader
import numpy as np

import tensorflow as tf
import inferpy as inf

datafolder = "./tests/files/"


@pytest.mark.parametrize("args, expected", [
        # single csv with header
    (
        dict(path=datafolder+"dataxy_with_header.csv"),
        1000
    ),
        # multiple csv files with header
    (
        dict(path=[datafolder + "dataxy_with_header.csv"]*4),
        4000
    ),
        # single csv without header
    (
        dict(path=datafolder + "dataxy_no_header.csv"),
        1000
    ),
])

def test_csv_size(args, expected):
    # build the data loader object
    data_loader = CsvLoader(**args)
    # assert that it is equal to expected
    assert data_loader.size == expected





def test_dict_size():
    # build the data loader object
    data_loader = SampleDictLoader({"a": np.random.rand(1000,2)})
    # assert that it is equal to expected
    assert data_loader.size == 1000




@pytest.mark.parametrize("args, expected", [
        # single csv with header
    (
        dict(path=datafolder+"dataxy_with_header.csv"),
        ["x", "y"]
    ),
        # multiple csv files with header
    (
        dict(path=[datafolder + "dataxy_with_header.csv"]*4),
        ["x", "y"]
    ),
        # single csv without header
    (
        dict(path=datafolder + "dataxy_no_header.csv"),
        ["0", "1"]
    ),
        # single csv with header and simple mapping
    (
        dict(path=datafolder + "dataxy_with_header.csv", var_dict={"x":[0], "y":[1]}),
        ["x", "y"]
    ),
        # single csv with header and grouping mapping
    (
        dict(path=datafolder + "dataxy_with_header.csv", var_dict={"a": [0,1]}),
        ["a"]
    ),
    # single csv with header and simple mapping
    (
        dict(path=datafolder + "dataxy_no_header.csv", var_dict={"x": [0], "y": [1]}),
        ["x", "y"]
    ),
])

def test_csv_dict_keys(args, expected):
    # build the data loader object
    data_loader = CsvLoader(**args)
    # assert that it is equal to expected
    assert set(data_loader.to_dict().keys()) == set(expected)





@pytest.mark.parametrize("data_loader, exp_keys", [
        # single csv with header
    (
        CsvLoader(path=datafolder+"dataxy_with_header.csv"),
        ["x", "y"]
    ),
        # multiple csv files with header
    (
        CsvLoader(path=[datafolder + "dataxy_with_header.csv"]*4),
        ["x", "y"]
    ),
        # single csv without header
    (
        CsvLoader(path=datafolder + "dataxy_no_header.csv"),
        ["0", "1"]
    ),
        # single csv with header and simple mapping
    (
        CsvLoader(path=datafolder + "dataxy_with_header.csv", var_dict={"x":[0], "y":[1]}),
        ["x", "y"]
    ),
        # single csv with header and grouping mapping
    (
        CsvLoader(path=datafolder + "dataxy_with_header.csv", var_dict={"a": [0,1]}),
        ["x", "y"]
    ),
    # single csv with header and simple mapping
    (
        CsvLoader(path=datafolder + "dataxy_no_header.csv", var_dict={"x": [0], "y": [1]}),
        ["0", "1"]
    ),
])

def test_batches(data_loader, exp_keys):

    batch = dict(data_loader.to_tfdataset(batch_size=50).make_one_shot_iterator().get_next())
    assert set(batch.keys()) == set(exp_keys)
    assert np.all([v.shape.as_list()[0] == 50 for v in batch.values()])







@pytest.mark.parametrize("data_loader, inf_method_name", [
        # single csv with header
    (
        CsvLoader(path=datafolder+"dataxy_with_header.csv"), "VI"
    ),
        # single csv with header and simple mapping
    (
        CsvLoader(path=datafolder + "dataxy_no_header.csv", var_dict={"x":[0], "y":[1]}), "VI"
    ),
    (
        CsvLoader(path=datafolder + "dataxy_with_header.csv"), "SVI"
    ),
    # single csv with header and simple mapping
    (
        CsvLoader(path=datafolder + "dataxy_no_header.csv", var_dict={"x": [0], "y": [1]}), "SVI"
    ),

])



def test_fit(data_loader, inf_method_name):
    @inf.probmodel
    def linear_reg(d):
        w0 = inf.Normal(0, 1, name="w0")
        w = inf.Normal(tf.zeros([d, 1]), 1, name="w")

        with inf.datamodel():
            x = inf.Normal(tf.ones([d]), 2, name="x")
            y = inf.Normal(w0 + x @ w, 1.0, name="y")

    @inf.probmodel
    def qmodel(d):
        qw0_loc = inf.Parameter(0., name="qw0_loc")
        qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
        qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

        qw_loc = inf.Parameter(tf.zeros([d, 1]), name="qw_loc")
        qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d, 1]), name="qw_scale"))
        qw = inf.Normal(qw_loc, qw_scale, name="w")

    # create an instance of the model
    m = linear_reg(d=1)

    vi = inf.inference.VI(qmodel(1), epochs=100)

    inf_method = getattr(inf.inference, inf_method_name)(qmodel(1), epochs=100)
    m.fit(data_loader, inf_method)




@pytest.mark.parametrize("data, expected_size, expected_type, exp_keys", [
        # single csv with header
    (
        CsvLoader([datafolder+"dataxy_with_header.csv"]),
        1000,
        "CsvLoader",
        ["x", "y"]

    ),
    # multiple csv with header
    (
            CsvLoader(3*[datafolder + "dataxy_with_header.csv"]),
            3*1000,
            "CsvLoader",
            ["x", "y"]
    ),
    # single csv with header - eager
    (
            CsvLoader([datafolder + "dataxy_with_header.csv"], force_eager=True),
            1000,
            "SampleDictLoader",
            ["x", "y"]
    ),
    # multiple csv with header - eager
    (
            CsvLoader(3 * [datafolder + "dataxy_with_header.csv"], force_eager=True),
            3 * 1000,
            "SampleDictLoader",
            ["x", "y"],
    ),
    ])
def test_build_data_loader(data, expected_size, expected_type, exp_keys):
    # build the data loader object
    data_loader = build_data_loader(data)

    assert data_loader.size == expected_size
    assert type(data_loader).__name__ == expected_type
    assert set(data_loader.variables) == set(exp_keys)



