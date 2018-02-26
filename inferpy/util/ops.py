import numpy as np
import inferpy.models


def get_total_dimension(x):

    D = 0

    if np.ndim(x) == 0:
        x = [x]

    for xi in x:
        if np.isscalar(xi):
            D = D + 1
        elif isinstance(xi, inferpy.models.RandomVariable):
            D = D + xi.dim
        else:
            raise ValueError("Wrong input type")


    return D