import tensorflow as tf
from tensorflow_probability import edward2 as ed


# this function is used to intercept the value property of edward2 random variables
def set_values(**model_kwargs):
    """Creates a value-setting interceptor. Usable as a parameter of the ed2.interceptor.

        :model_kwargs: The name of each argument must be the name of a random variable to intercept,
            and the value is the element which intercepts the value of the random variable.

        :returns: The random variable with the intercepted value
    """

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")

        # if name in model_kwargs, include the value as a new argument using model_kwargs[name]
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]

        randvar = ed.interceptable(f)(*args, **kwargs)

        # if the value used to intercept is an inferpy RandomVariable, assign to the object
        # the tf.Variables to observe the Random Variable as well
        if name in model_kwargs and \
                hasattr(model_kwargs[name], 'is_observed') and hasattr(model_kwargs[name], 'observed_value'):
            randvar.is_observed = model_kwargs[name].is_observed
            randvar.observed_value = model_kwargs[name].observed_value

        # finally return the built random variable
        return randvar

    return interceptor


# this function is used to intercept the value property of edward2 random variables
# dependent on a tf variable var_condition: if true use var_value, otherwise use the variable value
def set_values_condition(var_condition, var_value):
    """Creates a value-setting interceptor. Usable as a parameter of the ed2.interceptor.

        :var_condition (`tf.Variable`): The boolean tf.Variable, used to intercept the value property with
            `value_var` or the variable value property itself
        :var_value (`tf.Variable`): The tf.Variable used to intercept the value property when `var_condition` is True

        :returns: The random variable with the intercepted value
    """

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""

        # need to create the variable to obtain its value, and use it in the fn_false condition
        _value = ed.interceptable(f)(*args, **kwargs).value

        conditional_value = tf.cond(
            var_condition,
            lambda: var_value,
            lambda: _value
        )

        # need to broadcast to fix the shape of the tensor (which always be _value.shape)
        return ed.interceptable(f)(*args, value=tf.broadcast_to(conditional_value, _value.shape), **kwargs)

    return interceptor
