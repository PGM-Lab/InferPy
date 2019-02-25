import tensorflow as tf
import inspect

from . import loss_functions


class VI:
    def __init__(self, qmodel, loss='ELBO', optimizer='AdamOptimizer', epochs=1000):
        # TODO: implement qmodel automatic builder is qmodel is None

        if callable(qmodel):
            if len(inspect.signature(qmodel).parameters)>0:
                raise Exception("input qmodel can only be a callable object if this does not has any input parameter")
            self.qmodel = qmodel()
        else:
            self.qmodel


        if isinstance(loss, str):
            self.loss_fn = getattr(loss_functions, loss)
        else:
            self.loss_fn = loss

        self.epochs = epochs

        # Create optimizer if str (using default parameters)
        if isinstance(optimizer, str):
            self.optimizer = getattr(tf.train, optimizer)()
        else:
            self.optimizer = optimizer

    def run(self, pmodel, sample_dict):
        # Create expanded q model
        plate_size = pmodel._get_plate_size(sample_dict)
        qvars, qparams = self.qmodel.expand_model(plate_size)

        loss_tensor = self.loss_fn(pmodel, qvars, sample_dict)

        train = self.optimizer.minimize(loss_tensor)

        t = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.epochs):
                sess.run(train)

                t.append(sess.run(loss_tensor))
                if i % 200 == 0:
                    print("\n"+str(t[-1]), end="", flush=True)
                if i % 10 == 0:
                    print(".", end="", flush=True)

            # extract the inferred parameters
            params = {n: sess.run(p) for n, p in qparams.items()}

        return params
