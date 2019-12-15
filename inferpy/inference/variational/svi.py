import numpy as np
import tensorflow as tf

from inferpy import util
from inferpy import contextmanager
from .vi import VI
from inferpy.data.loaders import build_data_loader, DataLoader


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

    def compile(self, pmodel, data_size, extra_loss_tensor=None):
        # set the used pmodel
        self.pmodel = pmodel
        # compute the batch_weight depending on the data_size and the batch_size
        self.batch_weight = data_size / self.batch_size  # N/M
        # create the train tensor
        self.train_tensor = self._generate_train_tensor(extra_loss_tensor, batch_weight=self.batch_weight)

    def update(self, data):

        # create the input_data tensor
        data_loader = build_data_loader(data)
        input_data = self.create_input_data_tensor(data_loader)

        t = []
        sess = util.get_session()
        for i in range(self.epochs):
            for j in range(self.batches):
                # evaluate the data tensor to get an evaluated one which can be used to observe varoables
                local_input_data = sess.run(input_data)
                # reshape data in case it does not match exactly with the shape used when building the random variable
                # i.e.: (..., 1) dimension
                clean_local_input_data = {k: np.reshape(v, self.expanded_variables["p"][k].observed_value.shape.as_list())
                                          for k, v in local_input_data.items()}
                with contextmanager.observe(self.expanded_variables["p"], clean_local_input_data):
                    with contextmanager.observe(self.expanded_variables["q"], clean_local_input_data):
                        sess.run(self.train_tensor)

                        t.append(sess.run(self.debug.loss_tensor))
                        if j == 0 and i % 200 == 0:
                            print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
                        if j == 0 and i % 20 == 0:
                            print(".", end="", flush=True)

        # set the protected _losses attribute for the losses property
        self.debug.losses += t

    def create_input_data_tensor(self, data_loader):
        # NOTE: data_size, batches and batch_weight can be different in each iteration

        # create a tf dataset and an iterator, specifying the batch size
        data_size = data_loader.size
        self.batches = int(data_size / self.batch_size)  # M/N

        # ensure that the number of batches is equal or greater than 1
        if self.batches < 1:
            raise ValueError("The size of the data must be equal or greater than the batch size")

        data_loader.shuffle_buffer_size = data_size
        iterator = data_loader.to_tfdataset(self.batch_size).make_one_shot_iterator()

        # each time this tensor is evaluated in a session it contains new data
        input_data = data_loader.map_batch_fn(iterator.get_next())

        return input_data
