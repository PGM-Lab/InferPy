import tensorflow as tf
import itertools

import inferpy as inf
from inferpy import util
from inferpy import contextmanager
from .vi import VI


class SVI(VI):
    def __init__(self, *args, batch_size=100, **kwargs):
        # Build the super object
        super().__init__(*args, **kwargs)

        # and save the extra argument batch size
        self.batch_size = batch_size

    def run(self, pmodel, sample_dict):
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

        # Create the loss function tensor
        loss_tensor = self.loss_fn(pmodel, self.qmodel, plate_size=self.batch_size, batch_weight=batch_weight)

        train = self.optimizer.minimize(loss_tensor)

        t = []

        sess = inf.get_session()
        # Initialize all variables which are not in the probmodel p, because they have been initialized before
        model_variables = set([v for v in itertools.chain(
            pmodel.params.values(),
            (pmodel._last_expanded_params or {}).values(),
            (pmodel._last_fitted_params or {}).values(),
            self.qmodel.params.values(),
            (self.qmodel._last_expanded_params or {}).values(),
            (self.qmodel._last_fitted_params or {}).values()
            )])
        sess.run(tf.variables_initializer([v for v in tf.global_variables()
                                           if v not in model_variables and not v.name.startswith("inferpy-")]))

        for i in range(self.epochs):
            for j in range(batches):
                # evaluate the data tensor to get an evaluated one which can be used to observe varoables
                local_input_data = sess.run(input_data)
                with contextmanager.observe(pmodel._last_expanded_vars, local_input_data):
                    with contextmanager.observe(self.qmodel._last_expanded_vars, local_input_data):
                        sess.run(train)

                        t.append(sess.run(loss_tensor))
                        if i % 200 == 0:
                            print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
                        if i % 20 == 0:
                            print(".", end="", flush=True)

        # set the private __losses attribute for the losses property
        self.__losses = t

        return self.qmodel._last_expanded_vars, self.qmodel._last_expanded_params

