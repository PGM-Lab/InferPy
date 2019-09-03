import pytest
import os
from os.path import isfile, join

import inferpy as inf
import numpy as np
import tensorflow as tf



@pytest.mark.parametrize("pth, blacklist", [
    # ("./examples/docs/getting30s/", []),
    # ("./examples/docs/guidebayesian/", []),
     ("./examples/docs/guideinference/", []),
    # ("./examples/docs/guidemodels/", []),
    # ("./examples/docs/guidedata/", []),
])
def test_examples(pth, blacklist):
    #pth = "examples_old/"
    inf_examples = [f for f in os.listdir(pth) if isfile(join(pth, f)) and
                    f.endswith(".py") and not f.endswith("__init__.py") and f not in blacklist]

    failed = []

    for f in inf_examples:
        filename = join(pth, f)
        print("testing " + filename)
        try:
            exec(compile(open(filename, "rb").read(), filename, 'exec'))
        except Exception as e:
            failed.append(f)
            print("ERROR:")
            print(e)

    if len(failed) > 0:
        print("failed:")
        print(failed)

    assert len(failed) == 0

