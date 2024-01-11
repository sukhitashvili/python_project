import os.path
import unittest

from data_processors import TrainCSVProcessor, IdealCSVProcessor


class TestCSVProcessing(unittest.TestCase):
    def setUp(self):
        self.train_csv_location = "datasets/train.csv"
        self.ideals_csv_location = "datasets/ideal.csv"
        self.test_csv_location = "datasets/test.csv"
        self.tolerance = 0.1
        self.test_csv_file_locations()
        self.train_processor = TrainCSVProcessor(csv_path=self.train_csv_location)
        self.ideals_processor = IdealCSVProcessor(csv_path=self.ideals_csv_location)

    def test_csv_file_locations(self):
        """
        Check if csv files are located where they are expected to be by the default argparse arguments in main.py file.
        """
        self.assertTrue(os.path.isfile(self.train_csv_location))
        self.assertTrue(os.path.isfile(self.ideals_csv_location))
        self.assertTrue(os.path.isfile(self.test_csv_location))

    def test_x_column_equality(self):
        """
        Checks if X col values are equal in train and ideal csv files as the matching code logic assumes they are equal
        """
        sum_ = sum((self.train_processor['x'] - self.ideals_processor['x']) ** 2)
        self.assertEqual(sum_, 0)

    def test_train_matching_with_ideals(self):
        """
        Checks matching of columns from train.csv to ideals.csv is correct.
        It compares them to result which I got in homework.ipynb jupyter notebook.
        """
        res_dict = self.train_processor.match_to_ideals(ideals_df=self.ideals_processor.data)
        expected_dict = {'y1': 'y42', 'y2': 'y41', 'y3': 'y11', 'y4': 'y48'}  # values from homework.ipynb
        self.assertEqual(res_dict, expected_dict)  # check column matching

    def test_max_differences_to_ideals(self):
        """
        Checks if max differences between matched columns from train.csv to ideals.csv are approximately
        (tolerance=0.1) same as what I got in homework.ipynb jupyter notebook.
        The formula of calculating max differences was given in the assignment:
                                    max(abs(train_col - matched_col)) * np.sqrt(2)  (per matched columns)

        If all these tests are passed then assignment of test x,y pairs to ideal will be correct as well.
        """
        self.train_processor.match_to_ideals(ideals_df=self.ideals_processor.data)
        res_max_diffs = self.train_processor.find_max_differences_to_mappings(ideals_processor=self.ideals_processor)
        expected_max_diffs = {'y42': 0.7014046721030611,
                              'y41': 0.7038583326337785,
                              'y11': 0.7056020579561832,
                              'y48': 0.7067413342598947}  # values from homework.ipynb

        expected_sum_max_diffs = sum([i for i in expected_max_diffs.values()])
        res_max_diffs_sum = sum([i for i in res_max_diffs.values()])

        self.assertAlmostEqual(res_max_diffs_sum, expected_sum_max_diffs,
                               delta=self.tolerance)  # check if diffs are as expected


def suite():
    """
    Lists tests in predefined order of execution
    """
    suite = unittest.TestSuite()
    suite.addTest(TestCSVProcessing('test_csv_file_locations'))
    suite.addTest(TestCSVProcessing('test_x_column_equality'))
    suite.addTest(TestCSVProcessing('test_train_matching_with_ideals'))
    suite.addTest(TestCSVProcessing('test_max_differences_to_ideals'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
