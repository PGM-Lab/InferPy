# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from inferpy.util.session import get_session
import csv


class DataLoader:

    """ This class defines the basic functionality of any DataLoader """
    def __init__(self):
        raise NotImplementedError

    @property
    def size(self):
        """ Total number of instances in the data """
        return self._size

    @property
    def variables(self):
        """ List of variables over which is  the dataset defined"""
        return self._variables


    @property
    def map_batch_fn(self):
        """ Returns a function that transforms each tensor batch """
        if not self._map_batch_fn:
            return lambda x: x
        return self._map_batch_fn


    @map_batch_fn.setter
    def map_batch_fn(self, fn):
        """ Sets a function that transforms each tensor batch """
        self._map_batch_fn = fn

    @property
    def shuffle_buffer_size(self):
        """ Size of the shuffle size where 1 means no shuffle """
        return self._shuffle_buffer_size

    @shuffle_buffer_size.setter
    def shuffle_buffer_size(self, shuffle_buffer_size):
        """ Sets the size of the shuffle size where 1 implies no shuffle """
        self._shuffle_buffer_size = shuffle_buffer_size

    def to_tfdataset(self):
        """ Obtains a tensorflow dataset object"""
        raise NotImplementedError

    def to_dict(self):
        """ Obtains a dictionary with data as numpy objects"""
        raise NotImplementedError



class CsvLoader(DataLoader):
    """
    This class implements a data loader for datasets in CSV format
    """
    def __init__(self, path, var_dict=None, has_header=None, force_eager=False):
        """ Creates a new CsvLoader object

            Args:
                path (`str` or list of `str`): indicates the csv file(s) to load.
                var_dict (`dict`): mapping that associates each a variable name to a list
                    of integers indicating the columns in the file. The first column (excluding the
                    the tuple index) corresponds to 0.
                has_header (bool): indicates if the file has a header. If None, it will check it automatically.
                force_eager (`bool`): indicates if the data should always be loaded before the optimization
                    loop, regardless of the inference method.
        """

        if isinstance(path, str):
            path = [path]

        self._colnames = []
        self._size = 0
        self.has_header = None
        self._force_eager = force_eager

        for p in path:
            with open(p) as f:

                reader = csv.DictReader(f)

                if has_header is None:
                    has_header = csv.Sniffer().has_header(f.read(2048))
                    f.seek(0)

                # get the column names
                if has_header:
                    colnames = reader.fieldnames[1:]
                else:
                    colnames = [str(i) for i in range(len(reader.fieldnames[1:]))]

                if len(self._colnames)>0 and self._colnames != colnames:
                    raise ValueError("Error: header in csv files must be the same")

                if self.has_header != None and self.has_header != has_header:
                    raise ValueError("Error: header must either present or absent in all the csv files ")


                self._colnames = colnames
                self.has_header = has_header

                f.seek(0)
                self._size += sum(1 for line in f) - (1 if has_header else 0)

        self._path = path
        self._shuffle_buffer_size = 1

        if var_dict is None:
            var_dict = {self._colnames[i]: [i] for i in range(len(self._colnames))}

        self._map_batch_fn = self.__build_map_batch_fn(var_dict)
        self._variables = list(var_dict.keys())


    def __build_map_batch_fn(self, var_dict):
        """ This functions sets the property map_batch_fn with the
            function transforming each batch and consistent with the desired
            mapping.
        """
        def fn(batch):
            out_dict = {}
            for v, cols_idx in var_dict.items():
                cols = list(map(list(batch.values()).__getitem__, cols_idx))
                if len(cols)>1:
                    out_dict.update({v: tf.squeeze(tf.stack(cols, axis=1))})
                else:
                    out_dict.update({v:tf.expand_dims(cols[0], axis=1)})

            return out_dict
        return fn


    def to_tfdataset(self, batch_size = None):

        if batch_size == None: batch_size = self.size


        if self.has_header:
            col_args = {"select_columns": self._colnames}
        else:
            col_args = {"column_names": [""]+self._colnames,
                        "select_columns": list(range(1,len(self._colnames)+1))}


        # build the dataset object
        return tf.data.experimental.make_csv_dataset(self._path, batch_size=batch_size,
                                                     sloppy=True, shuffle=self.shuffle_buffer_size>1,
                                                     shuffle_buffer_size= self.shuffle_buffer_size,
                                                     **col_args
                                                    )


    def to_dict(self):
        return dict(get_session().run(
            self.map_batch_fn(
                self.to_tfdataset().make_one_shot_iterator().get_next()
            )
        ))




class SampleDictLoader(DataLoader):
    """
    This class implements a data loader for datasets in memory stored as dictionaries
    """
    def __init__(self, sample_dict):

        self.sample_dict = sample_dict

        # compute the size (and check the consistency)
        sizes = {tf.convert_to_tensor(col)._shape_as_list()[0] for col in sample_dict.values()}
        if len(sizes)>1:
            raise ValueError("Error: all the attributes in the sample_dict must have the same length")

        self._size = list(sizes)[0]
        self._map_batch_fn = None
        self._shuffle_buffer_size = 1
        self._variables = list(sample_dict.keys())


    def to_tfdataset(self, batch_size = None):

        if batch_size == None: batch_size = self.size

        return (
            tf.data.Dataset.from_tensor_slices(self.sample_dict)
                .shuffle(self.shuffle_buffer_size)
                .batch(batch_size)
                .repeat()
            )

    def to_dict(self):
        return self.sample_dict




def build_data_loader(data):
    """ This functions builds a DataLoader either from a dictionary or another
    DataLoader object """
    if isinstance(data, dict):
        data_loader = SampleDictLoader(data)
    elif isinstance(data, SampleDictLoader):
        data_loader = data
    elif isinstance(data, CsvLoader):
        if data._force_eager == False:
            data_loader = data
        else:
            data_loader = SampleDictLoader(data.to_dict())
    else:
        raise TypeError('The `data` type must be dict or DataLoader.')
    return data_loader


def build_sample_dict(data):
    """ This functions builds a dictionary either from other dictionary or from a
        DataLoader object """
    if isinstance(data, dict):
        data_loader = data
    elif isinstance(data, DataLoader):
        data_loader = data.to_dict()
    else:
        raise TypeError('The `data` type must be dict or DataLoader.')
    return data_loader



