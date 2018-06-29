import unittest
import inferpy as inf



class Probmodel_test_definition(unittest.TestCase):
    def test(self):
        N = 1
        d = 1

        with inf.replicate(size=N):
            c = inf.models.Categorical(probs=[0.4, 0.6], dim=d)
            m = inf.models.Multinomial(probs=[0.4, 0.2, 0.4], total_count=5, dim=d)
            n = inf.models.Normal(0, 1, dim=d)
            mn = inf.models.MultivariateNormalDiag(loc=[0, 0], scale_diag=[1, 1], dim=d)

        D = [c, m, n, mn]

        self.assertTrue([x.shape for x in D] == [[1], [1, 3], [1], [1, 2]])
        self.assertTrue([x.dim for x in D] == [1, 1, 1, 1])
        self.assertTrue([x.batches for x in D] == [1, 1, 1, 1])
        self.assertTrue([x.event_shape for x in D] == [[], [3], [], [2]])

        ############

        N = 10
        d = 1

        with inf.replicate(size=N):
            c = inf.models.Categorical(probs=[0.4, 0.6], dim=d)
            m = inf.models.Multinomial(probs=[0.4, 0.2, 0.4], total_count=5, dim=d)
            n = inf.models.Normal(0, 1, dim=d)
            mn = inf.models.MultivariateNormalDiag(loc=[0, 0], scale_diag=[1, 1], dim=d)

        D = [c, m, n, mn]

        self.assertTrue([x.shape for x in D] == [[10, 1], [10, 1, 3], [10, 1], [10, 1, 2]])
        self.assertTrue([x.dim for x in D] == [d * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.batches for x in D] == [N * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.event_shape for x in D] == [[], [3], [], [2]])

        ############


        N = 1
        d = 10

        with inf.replicate(size=N):
            c = inf.models.Categorical(probs=[0.4, 0.6], dim=d)
            m = inf.models.Multinomial(probs=[0.4, 0.2, 0.4], total_count=5, dim=d)
            n = inf.models.Normal(0, 1, dim=d)
            mn = inf.models.MultivariateNormalDiag(loc=[0, 0], scale_diag=[1, 1], dim=d)

        D = [c, m, n, mn]

        self.assertTrue([x.shape for x in D] == [[10], [10, 3], [10], [10, 2]])
        self.assertTrue([x.dim for x in D] == [d * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.batches for x in D] == [N * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.event_shape for x in D] == [[], [3], [], [2]])

        ############

        N = 10
        d = 5

        with inf.replicate(size=N):
            c = inf.models.Categorical(probs=[0.4, 0.6], dim=d)
            m = inf.models.Multinomial(probs=[0.4, 0.2, 0.4], total_count=5, dim=d)
            n = inf.models.Normal(0, 1, dim=d)
            mn = inf.models.MultivariateNormalDiag(loc=[0, 0], scale_diag=[1, 1], dim=d)

        D = [c, m, n, mn]

        self.assertTrue([x.shape for x in D] == [[10, 5], [10, 5, 3], [10, 5], [10, 5, 2]])
        self.assertTrue([x.dim for x in D] == [d * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.batches for x in D] == [N * i for i in [1, 1, 1, 1]])
        self.assertTrue([x.event_shape for x in D] == [[], [3], [], [2]])

        ##########


if __name__ == '__main__':
    unittest.main()


