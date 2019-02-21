from . import prob_model
from . import q_model


def get_active_model():
    # check if any context p or q model is active. Return such specific context, or None if any is active.
    if prob_model.is_active():
        return prob_model
    elif q_model.is_active():
        return q_model
    else:
        return None
