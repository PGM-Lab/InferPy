import unittest
import os
from os.path import isfile, join



class Test_examples(unittest.TestCase):
    def test(self):

        black_list = ["defmodels.py"]

        old_cwd = os.getcwd()
        ##os.chdir("../")   #uncomment in local

        #print(old_cwd+" changing to "+os.getcwd())

        pth = "examples/"
        inf_examples = [f for f in os.listdir(pth) if
                        isfile(join(pth, f)) and f.endswith(".py") and not f.endswith("__init__.py")]

        failed = []

        for f in inf_examples:

            if f not in black_list:

                filename = join(pth, f)
                print("testing " + filename)
                try:
                    exec (compile(open(filename, "rb").read(), filename, 'exec'))
                except Exception as e:
                    failed.append(f)
                    print("ERROR:")
                    print(e)


        #os.chdir(old_cwd)
        #print(old_cwd + " changing to " + os.getcwd())

        if len(failed) > 0:
            print("failed:")
            print(failed)
        self.assertTrue(len(failed)==0)



if __name__ == '__main__':
    unittest.main()


