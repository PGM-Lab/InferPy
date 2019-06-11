import tensorflow as tf

from inferpy import util
from inferpy import contextmanager
from .vi import VI


class SVI(VI):
    def __init__(self, *args, batch_size=100, **kwargs):
        """Creates a new Stochastic Variational Inference object.

            Args:
                *args: list of arguments used for the super().__init__ function
                *kwargs: dict of arguments used for the super().__init__ function
                batch_size (`int`): The number of epochs to run in the gradient descent process
        """
        # Build the super object
        super().__init__(*args, **kwargs)

        # and save the extra argument batch size
        self.batch_size = batch_size
        self.plate_size = batch_size  # the plate_size matches the batch_size
        self.batches = None
        self.batch_weight = None

    def compile(self, pmodel, data_size):
        # set the used pmodel
        self.pmodel = pmodel
        # compute the batch_weight depending on the data_size and the batch_size
        self.batch_weight = self.batch_size / data_size  # N/M
        # create the train tensor
        self.train_tensor = self._generate_train_tensor(batch_weight=self.batch_weight)

    def update(self, sample_dict):
        # create the input_data tensor
        input_data = self.create_input_data_tensor(sample_dict)

        t = []
        sess = util.get_session()
        for i in range(self.epochs):
            for j in range(self.batches):
                # evaluate the data tensor to get an evaluated one which can be used to observe varoables
                local_input_data = sess.run(input_data)
                with contextmanager.observe(self.expanded_variables["p"], local_input_data):
                    with contextmanager.observe(self.expanded_variables["q"], local_input_data):
                        sess.run(self.train_tensor)

                        t.append(sess.run(self.debug.loss_tensor))
                        if j == 0 and i % 200 == 0:
                            print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
                        if j == 0 and i % 20 == 0:
                            print(".", end="", flush=True)

        # set the protected _losses attribute for the losses property
        self.debug.losses += t

    def create_input_data_tensor(self, sample_dict):
        # NOTE: data_size, batches and batch_weight can be different in each iteration

        # create a tf dataset and an iterator, specifying the batch size
        data_size = util.iterables.get_plate_size(self.pmodel.vars, sample_dict)
        self.batches = int(data_size / self.batch_size)  # M/N

        # ensure that the number of batches is equal or greater than 1
        if self.batches < 1:
            raise ValueError("The size of the data must be equal or greater than the batch size")

        self.batch_weight = self.batch_size / data_size  # N/M

        tfdataset = (
            tf.data.Dataset.from_tensor_slices(sample_dict)
            .shuffle(data_size)  # use the size of the complete dataset for shuffle buffer, so we use a perfect shuffle
            .batch(self.batch_size, drop_remainder=True)  # discard the remainder batch with less elements if exists
            .repeat()
        )
        iterator = tfdataset.make_one_shot_iterator()
        input_data = iterator.get_next()  # each time this tensor is evaluated in a session it contains new data

        return input_data
