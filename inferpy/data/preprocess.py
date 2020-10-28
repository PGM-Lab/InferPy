import tensorflow as tf
import numpy as np
import math

from inferpy.util.session import get_session

def to_tensor(data, varnames=None):
    varnames = varnames or data.keys()
    return dict([
        (k,tf.convert_to_tensor(v) if k in varnames else v)
        for k,v in data.items()
    ])


def to_numpy(data, varnames=None):
    varnames = varnames or data.keys()

    # fn to convert a single value
    def convert_to_numpy(x):
        if isinstance(x, np.ndarray):
             return x
        if np.isscalar(x) or isinstance(x, list):
            return np.array(x)
        return convert_to_numpy(get_session().run(x))

    # convert all the values
    return dict([
        (k,convert_to_numpy(v) if k in varnames else v)
        for k,v in data.items()
    ])






def add_sample_dim(data, vars_datamodel):

    data_out = dict()
    for k,v in vars_datamodel.items():
        if k in data:
            data_out[k] = data[k]
            # data cannot be an scalar nor list
            if np.isscalar(data_out[k]) or isinstance(data_out[k], list):
                data_out[k] = np.array(data_out[k])

            # select the transforming gunctions depending on the type
            if isinstance(data_out[k], tf.Tensor):
                shape = data_out[k].get_shape().as_list()
                expand_dims = tf.expand_dims
                reshape = tf.reshape
            elif isinstance(data_out[k], np.ndarray):
                shape = data_out[k].shape
                expand_dims = np.expand_dims
                reshape = np.reshape
            else:
                raise ValueError("Unknown data type")

            var_shape = vars_datamodel[k].shape.as_list()

            # apply the transformation
            if len(var_shape) - len(shape) == 1:
                data_out[k] = expand_dims(data_out[k], axis=0)
            elif len(var_shape) - len(shape) == 2:
                data_out[k] = reshape(data_out[k], (1, 1))

    return data_out


def create_batches(data, vars_datamodel, data_size, batch_size, padding_last=True):
    data_out = data.copy()

    rows_to_add = round((math.ceil(data_size / batch_size) - data_size / batch_size) * batch_size)
    num_batches = math.ceil(data_size / batch_size)

    pad = {np.ndarray: np.pad, tf.Tensor:tf.pad}
    split = {np.ndarray: np.split, tf.Tensor:tf.split}

    # add padding if required
    if padding_last:
        for k, v in vars_datamodel.items():
            if k in data_out:
                padding = np.zeros([len(data_out[k].shape), 2], np.int32)
                padding[0, 1] = rows_to_add
                data_out[k] = pad[type(data_out[k])](data_out[k], padding, mode="constant")

    # create batches
    batches = [{} for _ in range(num_batches)]

    for k in data_out.keys():
        if k in vars_datamodel:
            batches_k = split[type(data_out[k])](data_out[k], num_batches, axis=0)
            for i in range(num_batches):
                batches[i][k] = batches_k[i]
        else:
            for i in range(num_batches):
                batches[i][k] = data_out[k]

    return batches
#
#
# data_ = data.copy()
#
#
#
#
#
# data_k.shape
#
#
# ###
# data = data_
#
# data["x"] = data["x"][0]
#
# vars_datamodel["z"].shape.as_list()
#
# data["x"] = 4
#
# data["x"].shape
# add_sample_dim(to_tensor(data), vars_datamodel)
#
# data = to_tensor(data_)
# data = data_