import tensorflow as tf
from tensorflow_probability import edward2 as ed


# this function is used to intercept the value property of edward2 random variables
def set_values(**model_kwargs):
    """Creates a value-setting interceptor."""

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]

        return ed.interceptable(f)(*args, **kwargs)

    return interceptor


# this function is used to intercept the value property of edward2 random variables
# dependent on a tf variable value: if true use predict_input dict, otherwise use variable value
def set_values_condition(condition_var, **input_dict):
    """Creates a value-setting interceptor."""

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.pop("name")

        _value = ed.interceptable(f)(*args, **kwargs).value

        conditional_value = tf.cond(
            tf.math.logical_and(condition_var, tf.constant(name in input_dict)),
            lambda: input_dict.get(name, 0.),  # ckecks that it return something, even if it is not used
            lambda: _value
        )

        return ed.interceptable(f)(*args, value=tf.broadcast_to(conditional_value, _value.shape), name=name, **kwargs)

    return interceptor
