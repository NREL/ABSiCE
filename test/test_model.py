from ABM_CE_PV_Model import ABM_CE_PV
import unittest

class TestABM_CE_PV(unittest.TestCase):
    def setUp(self):
        self.model = ABM_CE_PV()

    def test_tclp_test(self):
        # Test the tclp_test with samples of different ages
        hazardous_2000 = []
        hazardous_2010 = []
        for _ in range(1000):
            hazardous_2000.append(self.model.tclp_test(start_year=2000))
            hazardous_2010.append(self.model.tclp_test(start_year=2010))
        hazardous_2000_rate = sum(hazardous_2000) / len(hazardous_2000)
        hazardous_2010_rate = sum(hazardous_2010) / len(hazardous_2010)

        # Assert that older modules have a higher hazardous rate
        self.assertGreater(hazardous_2000_rate, hazardous_2010_rate)


if __name__ == '__main__':
    unittest.main()