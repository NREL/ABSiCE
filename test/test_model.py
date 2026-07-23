from ABM_CE_PV_Model import ABM_CE_PV
import unittest

class TestABM_CE_PV(unittest.TestCase):
    def setUp(self):
        self.model = ABM_CE_PV()

    def test_tclp_test(self):
        # Test the tclp_test with samples of different market-share years
        hazardous_2035 = []
        hazardous_2056 = []
        for _ in range(1000):
            hazardous_2035.append(self.model.tclp_test(
                self.model.tclp_market_share_df, current_year=2035, state='CA'))
            hazardous_2056.append(self.model.tclp_test(
                self.model.tclp_market_share_df, current_year=2056, state='CA'))
        hazardous_2035_rate = sum(hazardous_2035) / len(hazardous_2035)
        hazardous_2056_rate = sum(hazardous_2056) / len(hazardous_2056)

        # Assert that higher BSF share year has a higher hazardous rate
        self.assertGreater(hazardous_2035_rate, hazardous_2056_rate)


if __name__ == '__main__':
    unittest.main()