import unittest
import inferpy as inf



class Test_replicate(unittest.TestCase):
    def test(self):

        inf.replicate.delete_all()

        self.assertTrue(inf.replicate.in_replicate() == False)
        self.assertTrue(inf.replicate.get_active_replicate() == [])
        self.assertTrue(inf.replicate.get_all_replicate() == [])
        self.assertTrue(inf.replicate.get_total_size() == 1)

        with inf.replicate(size=5):


            print(inf.replicate.get_active_replicate()[-1].name)
            self.assertTrue(inf.replicate.in_replicate() == True)
            self.assertTrue(len(inf.replicate.get_active_replicate()) == 1)
            self.assertTrue(len(inf.replicate.get_all_replicate())==1)
            self.assertTrue(inf.replicate.get_total_size() == 5)

            with inf.replicate(size=5, name="A"):
                print(inf.replicate.get_active_replicate()[-1].name)
                self.assertTrue(inf.replicate.in_replicate() == True)
                self.assertTrue(len(inf.replicate.get_active_replicate()) == 2)
                self.assertTrue(len(inf.replicate.get_all_replicate()) == 2)
                self.assertTrue(inf.replicate.get_total_size() == 25)

            with inf.replicate(size=10, name="B"):
                print(inf.replicate.get_active_replicate()[-1].name)
                self.assertTrue(inf.replicate.in_replicate() == True)
                self.assertTrue(len(inf.replicate.get_active_replicate()) == 2)
                self.assertTrue(len(inf.replicate.get_all_replicate()) == 3)
                self.assertTrue(inf.replicate.get_total_size() == 50)

            with inf.replicate(name="A"):
                print(inf.replicate.get_active_replicate()[-1].name)
                with inf.replicate(name="B"):
                    print(inf.replicate.get_active_replicate()[-1].name)
                    self.assertTrue(inf.replicate.in_replicate() == True)
                    self.assertTrue(len(inf.replicate.get_active_replicate()) == 3)
                    self.assertTrue(len(inf.replicate.get_all_replicate()) == 3)
                    self.assertTrue(inf.replicate.get_total_size() == 250)


        with inf.replicate(name="A"):
            print(inf.replicate.get_active_replicate()[-1].name)
            with inf.replicate(name="B"):
                print(inf.replicate.get_active_replicate()[-1].name)
                self.assertTrue(inf.replicate.in_replicate() == True)
                self.assertTrue(len(inf.replicate.get_active_replicate()) == 2)
                self.assertTrue(len(inf.replicate.get_all_replicate()) == 3)
                self.assertTrue(inf.replicate.get_total_size() == 50)

                with inf.replicate(size=2):
                    print(inf.replicate.get_active_replicate()[-1].name)
                    self.assertTrue(inf.replicate.in_replicate() == True)
                    self.assertTrue(len(inf.replicate.get_active_replicate()) == 3)
                    self.assertTrue(len(inf.replicate.get_all_replicate()) == 4)
                    self.assertTrue(inf.replicate.get_total_size() == 100)

        self.assertTrue(inf.replicate.in_replicate() == False)
        self.assertTrue(len(inf.replicate.get_active_replicate()) == 0)
        self.assertTrue(len(inf.replicate.get_all_replicate()) == 4)
        self.assertTrue(inf.replicate.get_total_size() == 1)



if __name__ == '__main__':
    unittest.main()


