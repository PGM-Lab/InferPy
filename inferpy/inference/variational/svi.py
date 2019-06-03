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

    def run(self, pmodel, sample_dict):
        # set the used pmodel
        self.pmodel = pmodel

        # create a tf dataset and an iterator, specifying the batch size
        plate_size = util.iterables.get_plate_size(pmodel.vars, sample_dict)
        batches = int(plate_size / self.batch_size)  # M/N
        batch_weight = self.batch_size / plate_size  # N/M

        tfdataset = (
            tf.data.Dataset.from_tensor_slices(sample_dict)
            .shuffle(plate_size)  # use the size of the complete dataset for shuffle buffer, so we use a perfect shuffle
            .batch(self.batch_size, drop_remainder=True)  # discard the remainder batch with less elements if exists
            .repeat()
        )
        iterator = tfdataset.make_one_shot_iterator()
        input_data = iterator.get_next()  # each time this tensor is evaluated in a session it contains new data

        # Use self.batch_size as plate_size
        self.plate_size = self.batch_size

        # create the train tensor
        train = self._generate_train_tensor(batch_weight=batch_weight)

        t = []
        sess = util.get_session()
        for i in range(self.epochs):
            for j in range(batches):
                # evaluate the data tensor to get an evaluated one which can be used to observe varoables
                local_input_data = sess.run(input_data)
                with contextmanager.observe(self.expanded_variables["p"], local_input_data):
                    with contextmanager.observe(self.expanded_variables["q"], local_input_data):
                        sess.run(train)

                        t.append(sess.run(self.debug.loss_tensor))
                        if j == 0 and i % 200 == 0:
                            print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
                        if j == 0 and i % 20 == 0:
                            print(".", end="", flush=True)

        # set the protected _losses attribute for the losses property
        self.debug.losses = t
