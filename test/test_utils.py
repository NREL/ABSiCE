from utils import add_date_from_temporal_columns, TIMESTEP
import pandas as pd
import unittest

class TestUtils(unittest.TestCase):
    def test_add_date_from_temporal_columns(self):
        # Create a sample DataFrame with year, quarter, and month columns
        data = {
            'Year': [2020, 2020, 2020, 2020, 2021],
            'Quarter': [1, 2, 3, 4, 1],
            'Value': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)
        # Test with year and quarter columns
        result = add_date_from_temporal_columns(df.copy(), TIMESTEP.QUARTERLY)
        expected_dates = pd.Series(pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01', '2021-01-01']), name='date')
        pd.testing.assert_series_equal(result['date'], expected_dates)

    def test_add_date_from_temporal_columns_missing_columns(self):
        # Create a sample DataFrame missing the 'Quarter' column
        data = {
            'Year': [2020, 2020, 2020, 2020, 2021],
            'Value': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError) as context:
            add_date_from_temporal_columns(df.copy(), TIMESTEP.QUARTERLY)