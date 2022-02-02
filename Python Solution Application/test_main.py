import datetime
import os
from unittest import TestCase
from pathlib import Path
import shutil, tempfile

import pandas as pd

from main import *
import csv
import numpy as np

# FULL PATH OF THE SCRIPT DIRECTORY
path = Path(__file__).parent


class TestDataExtractor(TestCase):

    def setUp(self):
        # for testing environment

        # field names
        self.fields = ['Name', 'City']

        # data rows of csv file
        rows = [['Jack', 'CO'],
                ['Adam', 'WY'],
                ['Lisa', 'NC'],
                ['Sam', 'SC'],
                ['Pam', 'TX']]

        # name of csv file
        self.filename = "test.csv"

        # writing to csv file
        with open(self.filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(self.fields)

            # writing the data rows
            csvwriter.writerows(rows)

        csvObj = DataExtractor(path.joinpath(self.filename), self.fields)

        # dask dataframe
        self.data_frame = csvObj.csv_reader()

        # pandas dataframe
        self.pan_df = csvObj.dataframe_converter(self.data_frame)

    def tearDown(self):
        # Remove file after completing tests
        filePath = path.joinpath(self.filename)
        if os.path.exists(filePath):
            os.remove(filePath)

    def test_csv_reader_no_file(self):
        # if only path is given as an input
        test_DataExtractor = DataExtractor(path, [])
        with self.assertRaises(FileNotFoundError):
            test_DataExtractor.csv_reader()

    def test_csv_reader_different_file(self):
        # if file other than .csv is given as input
        test1_DataExtractor = DataExtractor(path.joinpath("blank"), [])
        with self.assertRaises(FileNotFoundError):
            test1_DataExtractor.csv_reader()

    def test_csv_reader_columns(self):
        # if correct columns appeared in dataframe
        expected_output = ['Name', 'City']
        column_names = self.data_frame.columns.values
        self.assertEqual(column_names.tolist(), expected_output)

    def test_csv_reader_length(self):
        # to check if length of dask dataframe equals length of csv file
        self.assertEqual(len(self.data_frame), 5)

    def test_dataframe_converter_length(self):
        # if both dataframes are of equal length
        self.assertEqual(len(self.data_frame), len(self.pan_df))

    def test_dataframe_converter_rowcheck(self):
        # to check if all expected rows are there
        expected_rows = ['Jack', 'Adam', 'Lisa', 'Sam', 'Pam']

        self.assertEqual(self.pan_df.iloc[:, 0].tolist(), expected_rows)

    def test_dataframe_converter_equal(self):
        # to check if dask to pandas conversion is true
        df = self.data_frame.compute()
        self.assertTrue(self.pan_df.equals(df))


class TestDataProcessor(TestCase):

    def setUp(self):
        # for testing environment
        self.dummy_string = "abcd"

        self.dummy_correct_str = "1990-05-05 12:17:23"

        """
        DATE_COLUMN_NAME = 'date'

        trans_col = "transaction_amount"
        COMMON_KEY_GROUP_BY = "A"
        """
        # Creating the first dataframe
        self.df1 = pd.DataFrame({"A": [1, 5, 7, 8],
                                 "B": [5, 8, 4, 3],
                                 "C": [10, 4, 9, 3]})

        # Creating the second dataframe
        self.df2 = pd.DataFrame({"A": [5, 3, 1, 1],
                                 "date": ["2021-08-31 21:15:17", "2021-09-05 06:44:27",
                                          "2021-12-05 06:24:27", "2021-10-05 06:27:17"],
                                 "transaction_amount": [15, 18, 14, 13]})

    """
    def tearDown(self):
        # Remove file after completing tests
        filePath = path.joinpath(self.filename)
        if os.path.exists(filePath):
            os.remove(filePath)
    """

    def test_string_to_timestamp_converter_wrong_string(self):
        # if wrong input string is given
        with self.assertRaises(ValueError):
            DataProcessor().string_to_timestamp_converter(self.dummy_string)

    def test_string_to_timestamp_converter_wrong_format(self):
        # wrong input format
        with self.assertRaises(TypeError):
            DataProcessor().string_to_timestamp_converter([])

    def test_string_to_timestamp_converter_correct_input(self):
        # to check if correct conversion happens between string and timestamp
        pd_dt = DataProcessor().string_to_timestamp_converter(self.dummy_correct_str)

        # convert timestamp back to string and check if both values match
        timestamp_to_string = pd_dt.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
        self.assertEqual(timestamp_to_string, self.dummy_correct_str)

    def test_dataframe_merger_correct_dfs(self):
        # to check if two dataframes are merged correctly
        result = DataProcessor().dataframe_merger(self.df1, self.df2, "A", "inner")
        expected_list = [1, 1, 5]
        self.assertEqual(result.iloc[:, 0].tolist(), expected_list)

    def test_dataframe_merger_correct_dfs_wrong_key(self):
        # when unknown key is provided as foreign key
        with self.assertRaises(KeyError):
            DataProcessor().dataframe_merger(self.df1, self.df2, "P", "inner")

    def test_dataframe_groupby_sum_wrong_key(self):
        # when unknown key is provided as foreign key
        with self.assertRaises(KeyError):
            DataProcessor().dataframe_groupby(self.df1, "P", "A", "sum", "col_name")

    def test_dataframe_groupby_sum_correct_case(self):
        # when all parameters are correct, output passes
        grouped = DataProcessor().dataframe_groupby(self.df2, "A", "transaction_amount", "sum", "amount")

        expected_row0 = [1, 3, 5]
        expected_row1 = [27, 18, 15]
        self.assertEqual(grouped.iloc[:, 0].tolist(), expected_row0)
        self.assertEqual(grouped.iloc[:, 1].tolist(), expected_row1)

    def test_dataframe_groupby_count_correct_case(self):
        # when all parameters for count are correct, output passes
        grouped = DataProcessor().dataframe_groupby(self.df2, "A", "transaction_amount", "count", "total")

        expected_row0 = [1, 3, 5]
        expected_row1 = [2, 1, 1]
        self.assertEqual(grouped.iloc[:, 0].tolist(), expected_row0)
        self.assertEqual(grouped.iloc[:, 1].tolist(), expected_row1)


class TestSQLQuery(TestCase):

    def setUp(self) -> None:
        # Creating the second dataframe
        self.test_df = pd.DataFrame({"hhid": [5, 3, 1, 1],
                                     "date": ["2021-08-31 21:15:17", "2021-09-07 06:44:27",
                                              "2021-12-08 06:24:27", "2021-10-05 06:27:17"],
                                     "transaction_amount": [15, 18, 14, 13]})

    def test_feature2_correctness(self):
        # all correct parameters are passed
        self.test_df["date"] = pd.to_datetime(self.test_df["date"])
        test2 = SQLQuery().feature2(self.test_df, "date",
                                    DataProcessor().string_to_timestamp_converter("2021-09-06 00:00:00"))
        self.assertTrue(test2.iloc[:, 0].tolist(), [5])

    def test_feature2_wrong_date_type(self):
        # when date parameter is not are passed

        with self.assertRaises(KeyError):
            SQLQuery().feature2(self.test_df, "A",
                                DataProcessor().string_to_timestamp_converter("2021-09-06 00:00:00"))

    def test_feature3_all_correct(self):
        self.test_df["date"] = pd.to_datetime(self.test_df["date"])
        lower = DataProcessor().string_to_timestamp_converter("2021-07-06 00:00:00")
        upper = DataProcessor().string_to_timestamp_converter("2021-12-06 00:00:00")
        test3 = SQLQuery().feature3(self.test_df, "date", lower, upper)

        self.assertTrue(test3.iloc[:, 1].tolist(), [13, 18, 15])

    def test_feature3_wrong_datetype(self):
        with self.assertRaises(TypeError):
            SQLQuery().feature3(self.test_df, "date", "abcd", "123")

    def test_feature3_wrong_col_name(self):
        self.test_df["date"] = pd.to_datetime(self.test_df["date"])
        lower = DataProcessor().string_to_timestamp_converter("2021-07-06 00:00:00")
        upper = DataProcessor().string_to_timestamp_converter("2021-12-06 00:00:00")

        with self.assertRaises(KeyError):
            SQLQuery().feature3(self.test_df, "P", lower, upper)

    def test_feature4_all_correct_parameters(self):
        test4 = SQLQuery().feature4(self.test_df, "hhid", "amount")
        expected = [2, 1, 1]
        self.assertTrue(test4.iloc[:, 1].tolist(), expected)

    def test_feature4_wrong_key(self):
        with self.assertRaises(KeyError):
            SQLQuery().feature4(self.test_df, "A", "amount")

    def test_feature1_all_correct_parameters(self):
        test1 = SQLQuery().feature1([self.test_df, self.test_df], "hhid")
        expected = [5, 3, 1, 1, 1, 1]
        self.assertEqual(test1.iloc[:, 0].tolist(), expected)

    def test_feature1_empty_list(self):
        with self.assertRaises(ValueError):
            SQLQuery().feature1([], "hhid")


class TestLoader(TestCase):

    def setUp(self) -> None:
        # create dataframe
        self.df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                                'Max Speed': [380., 370., 24., 26.]})

        self.filename = "output_test.csv"

    def tearDown(self) -> None:
        # Remove file after completing tests
        filePath = path.joinpath(self.filename)
        if os.path.exists(filePath):
            os.remove(filePath)

    def test_write_csv_empty_df(self):
        empty = pd.DataFrame()  # initializes an empty dataframe
        with self.assertRaises(ValueError):
            Loader().write_csv(empty, path.joinpath(self.filename))

    def test_write_csv(self):
        Loader().write_csv(self.df, path.joinpath(self.filename))

        expected = [['Animal', 'Max Speed'], ['Falcon', '380.0'], ['Falcon', '370.0'],
                    ['Parrot', '24.0'], ['Parrot', '26.0']]

        rows = []
        with open(self.filename, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)
        self.assertEqual(rows, expected)

    def test_write_csv_wrong_filetype(self):
        with self.assertRaises(FileNotFoundError):
            Loader().write_csv(self.df, path.joinpath(""))


class Test(TestCase):

    # Find the path from where the main.py file is running
    THIS_DIR = os.path.dirname(os.path.realpath(__file__))

    # join the parent directory with the main.py file name
    main_file_path = os.path.join(THIS_DIR, 'main.py')

    python_interpreter = 'python3.6'

    # generate command by taking the input output variables from main file
    command = python_interpreter + ' ' + main_file_path

    filename = "household_features.csv"

    def tearDown(self) -> None:
        # Remove file after completing tests
        filePath = path.joinpath(self.filename)
        if os.path.exists(filePath):
            os.remove(filePath)

    def test_main(self):

        # to check integration testing
        result = os.system(self.command)
        self.assertEqual(result, 0)
