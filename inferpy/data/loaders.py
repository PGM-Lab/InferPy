import tensorflow as tf

class DataLoader:

    """ This class defines the basic functionality of any DataLoader """
    def __init__(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    @property
    def tfdataset(self):
        return self._tfdataset




class CsvLoader(DataLoader):
    """
    This class implements a data loader for datasets in CSV format
    """
    def __init__(self, path, batch_size=1000, **kwargs):
        # build the dataset object
        self._tfdataset = tf.contrib.data.make_csv_dataset(path, batch_size, **kwargs)

        # compute the size
        with open(path) as f:
            self._size = sum(1 for line in f)



class SampleDictLoader(DataLoader):
    """
    This class implements a data loader for datasets in memory stored as dictionaries
    """
    def __init__(self, sample_dict):
        # build the dataset object
        self._tfdataset = tf.data.Dataset.from_tensor_slices(sample_dict)

        # compute the size (and check the consistency)
        sizes = {tf.convert_to_tensor(col)._shape_as_list()[0] for col in sample_dict.values()}
        if len(sizes)>1:
            raise ValueError("Error: all the attributes in the sample_dict must have the same length")

        self.size = sizes.pop()

