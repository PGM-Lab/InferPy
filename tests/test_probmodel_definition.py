import unittest

import inferpy as inf
from inferpy.models import Normal
import numpy as np


class Probmodel_test_definition(unittest.TestCase):
    def test(self):


        with inf.ProbModel() as m:
            x = Normal(loc=1., scale=100, name="x", observed=True)
            y = Normal(loc=x, scale=0.0001, dim=3, name="y")

        # print the list of variables
        print(m.varlist)
        print(m.latent_vars)
        print(m.observed_vars)

        # get a sample
        m_sample = m.sample()
        print("sample:")
        print(m_sample)

        self.assertTrue(np.abs(np.mean(list(m_sample.values())[0]-list(m_sample.values())[1])) < 1)


        # compute the log_prob for each element in the sample
        print(m.log_prob(m_sample))

        # compute the sum of the log_prob
        print(m.sum_log_prob(m_sample))

        self.assertTrue(len(m.varlist)==2)
        self.assertTrue(len(m.latent_vars)==1)
        self.assertTrue(len(m.latent_vars)==1)

        self.assertFalse(m.is_compiled())

        m.compile()

        self.assertTrue(m.is_compiled())

        z = Normal(loc=1., scale=1., dim=3, name="z")
        m.add_var(z)


        self.assertFalse(m.is_compiled())



if __name__ == '__main__':
    unittest.main()


