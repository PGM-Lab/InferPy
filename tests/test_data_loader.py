import pytest
from inferpy.data.loaders import CsvLoader, SampleDictLoader
import numpy as np

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



