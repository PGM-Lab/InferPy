import tensorflow as tf
import pandas as pd


class DataLoader:

    """ This class defines the basic functionality of any DataLoader """
    def __init__(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    @property
    def tfdataset(self):
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

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self,batch_size):
        self._batch_size = batch_size


class CsvLoader(DataLoader):
    """
    This class implements a data loader for datasets in CSV format
    """
    def __init__(self, path, variables, **kwargs):

        if isinstance(path, str):
            path = [path]

        self.variables = variables


        # get the column names
        self._colnames = pd.read_csv(path[0], index_col=0, nrows=0).columns.tolist()

        # compute the size
        self._size = 0
        for p in path:
            with open(p) as f:
                self._size += sum(1 for line in f) - 1      # -1 because of the header


        self._path = path
        self._batch_size = self.size
        self._shuffle_buffer_size = 1

    @property
    def tfdataset(self):

        # build the dataset object
        return tf.data.experimental.make_csv_dataset(self._path, batch_size=self.batch_size,
                                                     select_columns=self._colnames,
                                                     sloppy=True, shuffle=self.shuffle_buffer_size>1,
                                                     shuffle_buffer_size= self.shuffle_buffer_size
                                                    )




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

        self._batch_size = self.size
        self._shuffle_buffer_size = 1


    @property
    def tfdataset(self):
        return (
            tf.data.Dataset.from_tensor_slices(self.sample_dict)
                .shuffle(self.shuffle_buffer_size)
                .batch(self.batch_size)
                .repeat()
            )


