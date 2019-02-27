import tensorflow as tf
import inspect

from . import loss_functions


class VI:
    def __init__(self, qmodel, loss='ELBO', optimizer='AdamOptimizer', epochs=1000):
        # TODO: implement qmodel automatic builder is qmodel is None

        if callable(qmodel):
            if len(inspect.signature(qmodel).parameters) > 0:
                raise Exception("input qmodel can only be a callable object if this does not has any input parameter")
            self.qmodel = qmodel()
        else:
            self.qmodel = qmodel

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

    def run(self, pmodel, data):
        # NOTE: right now we use a session in a with context, so it is open and close.
        # If we want to use consecutive inference, we need the same session to reuse the same variables.
        # In this case, the build_in_session function from RandomVariables should not be used.

        # Create the loss function tensor
        loss_tensor = self.loss_fn(pmodel, self.qmodel, data)

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
            params = {n: sess.run(p) for n, p in self.qmodel._last_expanded_params.items()}

            posterior_qvars = {name: qv.build_in_session(sess) for name, qv in self.qmodel._last_expanded_vars.items()}

        return posterior_qvars, params
