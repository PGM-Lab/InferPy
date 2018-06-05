import inferpy as inf
import edward as ed
from six import iteritems

ALLOWED_METRICS = ['binary_accuracy',
'categorical_accuracy',
'sparse_categorical_accuracy',
'log_loss',
'binary_crossentropy',
'categorical_crossentropy',
'sparse_categorical_crossentropy',
'hinge',
'squared_hinge',
'mse',
'MSE',
'mean_squared_error',
'mae',
'MAE',
'mean_absolute_error',
'mape',
'MAPE',
'mean_absolute_percentage_error',
'msle',
'MSLE',
'mean_squared_logarithmic_error',
'poisson',
'cosine',
'cosine_proximity',
'log_lik',
'log_likelihood']


def evaluate(metrics, data, n_samples=500, output_key=None, seed=None):

    data_ed = {}

    for (key, value) in iteritems(data):
        data_ed.update(
            {key.dist if isinstance(key, inf.models.RandomVariable) else key :
                 value.dist if isinstance(value, inf.models.RandomVariable) else value})

    output_key_ed = output_key.dist if isinstance(output_key, inf.models.RandomVariable) else output_key

    return ed.evaluate(metrics, data_ed, n_samples, output_key_ed, seed)