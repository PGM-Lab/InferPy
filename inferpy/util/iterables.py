
def get_shape(x):
    """
    Get the shape of an element x. If it is an element with a shape attribute, return it. If it is a list with more than
    one element, compute the shape by checking the len, and the shape of internal elements. In that case, the shape must
    be consistent. Finally, in other case return () as shape.

    :param x: The element to compute its shape
    :raises : class `ValueError`: list shape not consistent
    :returns: A tuple with the shape of `x`
    """
    if isinstance(x, list) and len(x) > 0:
        shapes = [get_shape(subx) for subx in x]
        if any([s != shapes[0] for s in shapes[1:]]):
            raise ValueError('Parameter dimension not consistent: {}'.format(x))
        return (len(x), ) + shapes[0]
    else:
        if hasattr(x, 'shape'):
            return tuple(x.shape)
        else:
            return ()
