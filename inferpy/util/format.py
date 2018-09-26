# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


""" Module implementing text formatting operaionts """




import numpy as np
from inferpy.util.runtime import Runtime

def np_str(s):

    """ Shorten string representation of a numpy object

        Args:
            s: numpy object.

    """

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
