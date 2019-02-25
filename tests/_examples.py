import os
from os.path import isfile, join


# List of files that we do not want to test
BLACK_LIST = ["defmodels.py"]


def test_examples():
    pth = "examples_old/"
    inf_examples = [f for f in os.listdir(pth) if isfile(join(pth, f)) and
                    f.endswith(".py") and not f.endswith("__init__.py") and f not in BLACK_LIST]

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
