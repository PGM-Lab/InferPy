import tensorflow as tf
from tensorflow_probability import edward2 as ed
from contextlib import contextmanager
from inferpy import util


# Global variable to access when enable_interceptor is used. However, the vaiable will be None in the finally clause,
# so a local variable needs to be used in the set_value function, and the real enable_variable should never be deleted
# so the local variable always point to the good one.
CURRENT_ENABLE_INTERCEPTOR = None

# global variable which allow to use or not conditions, independently of CURRENT_ENABLE_INTERCEPTOR value
ALLOW_CONDITIONS = True


@contextmanager
def disallow_conditions():
    global ALLOW_CONDITIONS
    ALLOW_CONDITIONS = False
    try:
        yield
    finally:
        ALLOW_CONDITIONS = True


@contextmanager
def enable_interceptor(enable_variable):
    global CURRENT_ENABLE_INTERCEPTOR
    sess = util.session.get_session()
    try:
        if enable_variable:
            enable_variable.load(True, session=sess)
            CURRENT_ENABLE_INTERCEPTOR = enable_variable
        yield
    finally:
        if enable_variable:
            enable_variable.load(False, session=sess)
            CURRENT_ENABLE_INTERCEPTOR = None


# this function is used to intercept the value property of edward2 random variables
def set_values(**model_kwargs):  # noqa: E999
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
            interception_value = model_kwargs[name]

            if ALLOW_CONDITIONS and CURRENT_ENABLE_INTERCEPTOR is not None:
                # local variable points to the real condition variable (created in the inference method object)
                # this way, even if CURRENT_ENABLE_INTERCEPTOR is set to None, this local variable points to the real one
                condition_variable = CURRENT_ENABLE_INTERCEPTOR

                # need to create the variable to obtain its value, and use it in the fn_false condition
                _value = ed.interceptable(f)(*args, **kwargs).value

                conditional_value = tf.cond(
                    condition_variable,
                    lambda: interception_value,
                    lambda: _value
                )

                # need to broadcast to fix the shape of the tensor (which always be _value.shape)
                kwargs['value'] = tf.broadcast_to(conditional_value, _value.shape)
            else:
                kwargs['value'] = interception_value

            return ed.interceptable(f)(*args, **kwargs)
        else:
            return ed.interceptable(f)(*args, **kwargs)

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

        if ALLOW_CONDITIONS:
            # need to create the variable to obtain its value, and use it in the fn_false condition
            _value = ed.interceptable(f)(*args, **kwargs).value

            conditional_value = tf.cond(
                var_condition,
                lambda: var_value,
                lambda: _value
            )

            # need to broadcast to fix the shape of the tensor (which always be _value.shape)
            return ed.interceptable(f)(*args, value=tf.broadcast_to(conditional_value, _value.shape), **kwargs)
        else:
            return ed.interceptable(f)(*args, **kwargs)

    return interceptor


def make_predictable_variables(initial_value, rv_name):
    if ALLOW_CONDITIONS:
        is_observed = tf.Variable(False, trainable=False,
                                  name="inferpy-predict-enabled-{name}".format(name=rv_name or "default"))

        observed_value = tf.Variable(initial_value, trainable=False,
                                     name="inferpy-predict-{name}".format(name=rv_name or "default"))

        util.session.get_session().run(tf.variables_initializer([is_observed, observed_value]))

        return is_observed, observed_value
    else:
        return None, None
