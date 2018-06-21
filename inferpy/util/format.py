import numpy as np
from inferpy.util.runtime import Runtime

def np_str(s):

    if not Runtime.compact_param_str:
        return str(s)

    old_ops = np.get_printoptions()
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=5)

    isnp = isinstance(s, np.ndarray)
    out = np.array_repr(s) if isnp else str(s)



    np.set_printoptions(precision=old_ops.get('precision'))
    np.set_printoptions(threshold=old_ops.get('threshold'))

    try:
        i = out.index(',')
    except ValueError:
        i = 0



    try:
        j = out.rindex('...')
    except ValueError:
        j = - 1


    if j>0:

        out = out[0:i] + "," + out[j:-1]

        try:
            k = out.rindex(',')
        except ValueError:
            k = len(out) - 1

        out = out[0:k]


    if isnp:
        out = out.replace("array(", "")


    return out
