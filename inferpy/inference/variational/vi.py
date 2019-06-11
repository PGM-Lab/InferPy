import tensorflow as tf
import inspect
import itertools
from tensorflow_probability.python import edward2 as ed

from . import loss_functions
import inferpy as inf
from inferpy import util
from inferpy import contextmanager

from ..inference import Inference


class VI(Inference):
    def __init__(self, qmodel, loss='ELBO', optimizer='AdamOptimizer', epochs=1000):
        """Creates a new Variational Inference object.

            Args:
                qmodel (`inferpy.ProbModel`): The q model
                loss (`str` or `function`): A function that computes the loss tensor from the expanded variables
                    of p and q models, or a `str` that refears to a function with that name in the package
                    `inferpy.inference.variational.loss_function`
                optimizer (`str` or `tf.train.Optimizer`): An optimizer object from `tf.train` optimizers, or a string
                    that refers to the name of an optimizer in such module or package
                epochs (`int`): The number of epochs to run in the gradient descent process
        """

        # store the qmodel in self.qmodel. Can be a callable with no parameters which returns the qmodel
        if callable(qmodel):
            if len(inspect.signature(qmodel).parameters) > 0:
                raise ValueError("input qmodel can only be a callable object if this does not has any input parameter")
            self.qmodel = qmodel()
        else:
            self.qmodel = qmodel

        # store the loss function in self.loss_fn
        # if it is a string, build the object automatically from loss_functions package
        if isinstance(loss, str):
            self.loss_fn = getattr(loss_functions, loss)
        else:
            self.loss_fn = loss

        self.epochs = epochs

        # store the optimizer function in self.optimizer
        # if it is a string, build a new optimizer from tf.train (default parametrization)
        if isinstance(optimizer, str):
            self.optimizer = getattr(tf.train, optimizer)()
        else:
            self.optimizer = optimizer

        # pmodel not established yet
        self.pmodel = None
        # The size of the plate when expand the models
        self.plate_size = None
        # The tensor to optimize the tf.Variables
        self.train_tensor = None

        # expanded variables and parameters
        self.expanded_variables = {"p": None, "q": None}
        self.expanded_parameters = {"p": None, "q": None}

        # list for storing the loss evolution
        class Debug:
            pass
        self.debug = Debug()

        self.debug.losses = []

    def compile(self, pmodel, data_size):
        # set the used pmodel
        self.pmodel = pmodel
        # and the plate size, which matches the data size
        self.plate_size = data_size
        # create the train tensor
        self.train_tensor = self._generate_train_tensor(plate_size=self.plate_size)

    def update(self, sample_dict):
        # ensure that the size of the data matches with the self.plate_size
        data_size = util.iterables.get_plate_size(self.pmodel.vars, sample_dict)
        if data_size != self.plate_size:
            raise ValueError("The size of the data must be equal to the plate size: {}".format(self.plate_size))

        t = []
        sess = util.get_session()
        with contextmanager.observe(self.expanded_variables["p"], sample_dict):
            with contextmanager.observe(self.expanded_variables["q"], sample_dict):
                for i in range(self.epochs):
                    sess.run(self.train_tensor)

                    t.append(sess.run(self.debug.loss_tensor))
                    if i % 200 == 0:
                        print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
                    if i % 10 == 0:
                        print(".", end="", flush=True)

        # set the protected _losses attribute for the losses property
        self.debug.losses += t

    @property
    def losses(self):
        return self.debug.losses

    def sample(self, size=1, data={}):
        raise NotImplementedError

    def log_prob(self, data):
        raise NotImplementedError

    def parameters(names=None):
        raise NotImplementedError

    def _generate_train_tensor(self, **kwargs):
        """ This function expand the p and q models. Then, it uses the  loss function to create the loss tensor
            and store it into the debug object as a new attribute.
            Then, uses the optimizer to create the train tensor used in the gradient descent iterative process.
            It store the expanded random variables and parameters from the p and q models in self.expanded_variables
            and self.expanded_parameters dicts.

            Returns:
                The `tf.Tensor` train tensor used in the gradient descent iterative process.

        """
        # expand the p and q models
        # expand de qmodel
        qvars, qparams = self.qmodel.expand_model(self.plate_size)

        # expand de pmodel, using the intercept.set_values function, to include the sample_dict and the expanded qvars
        with ed.interception(util.interceptor.set_values(**qvars)):
            pvars, pparams = self.pmodel.expand_model(self.plate_size)

        # create the loss tensor and trainable tensor for the gradient descent process
        loss_tensor = self.loss_fn(pvars, qvars, **kwargs)
        train = self.optimizer.minimize(loss_tensor)

        # save the expanded variables and parameters
        self.expanded_variables = {
            "p": pvars,
            "q": qvars
        }
        self.expanded_parameters = {
            "p": pparams,
            "q": qparams
        }
        # save the loss tensor for debug purposes
        self.debug.loss_tensor = loss_tensor

        # Initialize all variables which are not in the probmodel p, because they have been initialized before
        model_variables = [v for v in itertools.chain(
            self.pmodel.params.values(),  # do not re-initialize prior p model parameters
            pparams.values(),  # do not re-initialize posterior p model parameters
            self.qmodel.params.values(),  # do not re-initialize prior q model parameters
            qparams.values(),  # do not re-initialize posterior q model parameters
           )]
        inf.get_session().run(
            tf.variables_initializer([
                v for v in tf.global_variables() if v not in model_variables and not v.name.startswith("inferpy-")
                ]))

        return train
