import tensorflow as tf
from inferpy.util.session import get_session
import csv


class DataLoader:

    """ This class defines the basic functionality of any DataLoader """
    def __init__(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    def to_tfdataset(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @property
    def map_batch_fn(self):
        if not self._map_batch_fn:
            return lambda x: x
        return self._map_batch_fn


    @map_batch_fn.setter
    def map_batch_fn(self, fn):
        self._map_batch_fn = fn


    @property
    def shuffle_buffer_size(self):
        return self._shuffle_buffer_size

    @shuffle_buffer_size.setter
    def shuffle_buffer_size(self, shuffle_buffer_size):
        self._shuffle_buffer_size = shuffle_buffer_size


#path = ['./playground_ignored/datalr.csv']

#path = ['./playground_ignored/datalr_nh.csv']

class CsvLoader(DataLoader):
    """
    This class implements a data loader for datasets in CSV format
    """
    def __init__(self, path, var_dict=None, **kwargs):

        if isinstance(path, str):
            path = [path]

        self._colnames = []
        self._size = 0
        self.has_header = None

        for p in path:
            with open(p) as f:

                reader = csv.DictReader(f)
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
        self.variables = var_dict.keys()


    def __build_map_batch_fn(self, var_dict):
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
        self.variables = sample_dict.keys()


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
    if isinstance(data, dict):
        data_loader = SampleDictLoader(data)
    elif isinstance(data, DataLoader):
        data_loader = data
    else:
        raise TypeError('The `data` type must be dict or DataLoader.')
    return data_loader


def build_sample_dict(data):
    if isinstance(data, dict):
        data_loader = data
    elif isinstance(data, DataLoader):
        data_loader = data.to_dict()
    else:
        raise TypeError('The `data` type must be dict or DataLoader.')
    return data_loader



