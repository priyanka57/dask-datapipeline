"""

Python script to generate one dataset at the household level by using input datasets.
Input datasets: demographics, transactions, household-individual mapping
Output table: will be saved in .CSV format

Author: Priyanka Goyal

"""
import datetime
from pathlib import Path
from dask import dataframe as dd
from functools import reduce
import pandas as pd

# FULL PATH OF THE SCRIPT DIRECTORY
path = Path(__file__).parent

# INPUT FILE NAMES
HH_FILENAME = "hh_ind.csv"
DEM_FILENAME = "demographics.csv"
TRANS_FILENAME = "transactions.csv"

# NECESSARY COLUMNS FOR DATAFRAME CONSTRUCTION
HH_COLS = ['hhid', 'individual_id']
DEM_COLS = ['hhid', 'num_inds', 'children_ind', 'hh_income_ind', 'age_ind', 'home_value_ind', 'state']
TRANS_COLS = ['individual_id', 'date', 'transaction_amount']

# campaign period
CAMPAIGN_START_DATE = '2021-09-06 00:00:00'
CAMPAIGN_END_DATE = '2021-09-13 23:59:59'

# TO CREATE FEATURES
COMMON_KEY_INNER_MERGE = 'individual_id'
COMMON_KEY_GROUP_BY = 'hhid'
ORIENTATION = 'inner'
DATE_COLUMN_NAME = 'date'
TRANS_AMOUNT_COL_NAME = 'transaction_amount'
AGGREGATION_TYPE = ["sum", "count"]
BEFORE_CAMPAIGN_SUM_COL_NAME = 'total_amount_before_campaign'
DURING_CAMPAIGN_SUM_COL_NAME = 'total_amount_during_campaign'
COUNT_COL_NAME = 'total_transactions'
CSV_FILENAME = 'household_features.csv'

# URLPATH FOR INPUT FILES
hh_filepath = path.joinpath(HH_FILENAME)
dem_filepath = path.joinpath(DEM_FILENAME)
trans_filepath = path.joinpath(TRANS_FILENAME)

# URLPATH FOR OUTPUT FILE
csv_filepath = path.joinpath(CSV_FILENAME)


class DataExtractor:
    """
    Class to intake a csv file and extract dataframe to
    build features
    """

    def __init__(self, input_filepath: Path, read_columns: list):
        """
        :param input_filepath: must end in .csv extension
        :param read_columns: list of column names to be read for each csv file
        """
        self.input_filepath = input_filepath
        self.read_columns = read_columns

    def csv_reader(self) -> dd.DataFrame:
        """
        This function will take a csv file and read it using DASK library
        for fast reading of bigdata files.
        :return: Dask DataFrame
        """
        if self.input_filepath.suffix == '.csv':
            dask_df = dd.read_csv(self.input_filepath, usecols=self.read_columns, dtype={'individual_id': 'str'})
        else:
            raise FileNotFoundError("Wrong file format, please input a .csv file")
        return dask_df

    def dataframe_converter(self, d_dataframe: dd.DataFrame) -> pd.DataFrame:
        """
        This function will convert DASK dataframe into a PANDAS dataframe to
        aid in faster processing of data
        :param d_dataframe: type dd.DataFrame
        :return: p_dataframe: type pd.DataFrame
        """
        if len(d_dataframe.index) == 0:
            raise ValueError("Empty dask dataframe")

        if not isinstance(d_dataframe, dd.DataFrame):
            raise TypeError("Expected Dask dataframe type")
        else:
            p_dataframe = d_dataframe.compute()
        return p_dataframe


class DataProcessor:
    """
    This class will act as a staging area to process the required data
    """

    def __init__(self):
        pass

    def string_to_timestamp_converter(self, date_str: str) -> pd.Timestamp:
        """
        Function to convert date string parameters to Timestamp
        :param date_str: string containing datetime stamp
        :return pd_timestamp: Pandas Timestamp object
        """
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError("Incorrect data format, should be %Y-%m-%d %H:%M:%S")

        if isinstance(date_str, str):
            pd_timestamp = pd.to_datetime(date_str, errors='coerce')
        else:
            raise TypeError("Enter string type in the format %Y-%m-%d %H:%M:%S")
        return pd_timestamp

    def dataframe_merger(self, df1: pd.DataFrame, df2: pd.DataFrame, common_key: str, orientation: str) -> pd.DataFrame:
        """
        Function to merge two pandas dataframes if they are not empty
        :param df1: input dataframe 1
        :param df2: input dataframe 2
        :param common_key: key to match in both dataframes
        :param orientation: which type of join
        :return df_merged: a merged pandas dataframe
        """
        if common_key not in df1.columns | df2.columns:  # bitwise or = '|'
            raise KeyError("{} column not found in the dataframe".format(common_key))

        df_merged = df1.merge(df2, on=common_key, how=orientation)

        # Convert date column to a pandas series

        df_merged[DATE_COLUMN_NAME] = pd.to_datetime(df_merged[DATE_COLUMN_NAME])

        return df_merged

    def dataframe_groupby(self, p_df: pd.DataFrame, groupby_col: str, transaction_col_name: str,
                          aggregation_type: str, new_column_name: str) -> pd.DataFrame:
        """
        Function to aid in feature2 and feature3 development.
        It will group by dataframe and rename the aggregation column
        :param aggregation_type: what type of aggregation is expected during GROUP BY
        :param transaction_col_name: Column name to be transformed
        :param p_df: input dataframe
        :param groupby_col: column which needs to be grouped
        :param new_column_name: renaming column name after performing aggregation
        :return df_grouped: return a grouped by dataframe
        """
        if not isinstance(p_df, pd.DataFrame):
            raise TypeError("Wrong data type, enter Pandas dataframe")

        if p_df.empty:
            raise ValueError("Empty pandas dataframe.")

        if groupby_col not in p_df.columns:
            raise KeyError("{} column not found in the dataframe".format(groupby_col))
        else:
            # only pick necessary columns
            p_df = p_df[[groupby_col, transaction_col_name]]

            # group by according to aggregation type and round off value 2 decimal points
            df_grouped = p_df.groupby(groupby_col, as_index=False).agg(aggregation_type).round(2)

            # rename aggregated column according to business requirement
            df_grouped.rename(columns={transaction_col_name: new_column_name}, inplace=True)

            # trial
            # df_grouped = p_df.groupby(groupby_col).transaction_amount.sum().rename(new_column_name).round(2)
        return df_grouped


class SQLQuery(DataProcessor):
    """
    Class to create all the required features at house hold level
    """

    def __init__(self):
        super().__init__()

    def feature2(self, f2_df: pd.DataFrame, column_date_name: str, date_before: pd.Timestamp) -> pd.DataFrame:
        """,
        Function to develop feature #2:
        At household level, total amount of dollars spent before the campaign period
        :param f2_df: input dataframe
        :param column_date_name: name of the date column in the dataframe
        :param date_before: provide datetime which starts before the campaign period
        :return f2_table: data table containing 'hhid' and associated dollars spent before requested time
        """
        if column_date_name not in f2_df.columns:
            raise KeyError("{} column not found in the dataframe".format(column_date_name))

        f2_table = f2_df[f2_df[column_date_name] < date_before]
        f2_table = self.dataframe_groupby(f2_table, COMMON_KEY_GROUP_BY, TRANS_AMOUNT_COL_NAME,
                                          AGGREGATION_TYPE[0], BEFORE_CAMPAIGN_SUM_COL_NAME)
        return f2_table

    def feature3(self, f3_df: pd.DataFrame, date_column_name: str,
                 lower_date: pd.Timestamp, upper_date: pd.Timestamp) -> pd.DataFrame:
        """
        Function to develop feature #3:
        At household level, total amount of dollars spent during the campaign period
        :param f3_df: input dataframe
        :param date_column_name: name of the date column in the dataframe
        :param lower_date: provide datetime when campaign period ended
        :param upper_date: provide datetime when campaign period ended
        :return f3_table: data table containing 'hhid' and associated dollars spent during requested time
        """

        if date_column_name not in f3_df.columns:
            raise KeyError("{} column not found in the dataframe".format(date_column_name))

        if not isinstance(lower_date, pd.Timestamp) or not isinstance(upper_date, pd.Timestamp):
            raise TypeError("Required pandas timestamp format date time")

        f3_table = f3_df[f3_df[date_column_name].between(lower_date, upper_date, inclusive=True)]
        f3_table = self.dataframe_groupby(f3_table, COMMON_KEY_GROUP_BY,
                                          TRANS_AMOUNT_COL_NAME, AGGREGATION_TYPE[0], DURING_CAMPAIGN_SUM_COL_NAME)
        return f3_table

    def feature4(self, f4_df: pd.DataFrame, groupby_column: str, new_col_name: str) -> pd.DataFrame:
        """
        Function to develop feature #4:
        At household level, total number of transactions
        :param f4_df: input dataframe
        :param groupby_column: column on which groupby will occur
        :param new_col_name: new name for the count column
        :return f4_table: data table containing 'hhid' and total number of transactions
        """

        if groupby_column not in f4_df.columns:
            raise KeyError("{} column not found in the dataframe".format(groupby_column))

        # calling dataframe_groupby on count aggregation
        f4_table = self.dataframe_groupby(f4_df, COMMON_KEY_GROUP_BY,
                                          TRANS_AMOUNT_COL_NAME, AGGREGATION_TYPE[1], new_col_name)

        # f4_table = f4_df.groupby(groupby_column).transaction_amount.count().rename(new_col_name)
        return f4_table

    def feature1(self, df_list: list, key_common: str) -> pd.DataFrame:
        """
        Function to develop feature #1:
        At household level, all the demographic features provided
        :param key_common: common key column on which all the dataframes will be merged
        :param df_list: list containing all the dataframes to merge
        :return final_df: final table containing household ids and their associated demographics
        """

        if not df_list:
            raise ValueError("Dataframe list is empty")
        else:
            final_df = reduce(lambda left, right: pd.merge(left, right, on=key_common), df_list)
        return final_df


class Loader:
    def __init__(self):
        pass

    def write_csv(self, output_df: pd.DataFrame, output_filepath: Path) -> bool:
        """
        Write final table to an output csv file
        :param output_filepath: name of the output file
        :param output_df: the final feature1 table to csv file
        :return bool: if file writes then True
        """

        if output_df.empty:
            raise ValueError("Output dataframe not found. Couldn't write output to a file.")

        if output_filepath.suffix == '.csv':
            output_df.to_csv(output_filepath, encoding='utf-8', index=False)
        else:
            raise FileNotFoundError("No such file or directory")

        return True


def main():
    # DataExtractor initialization
    hh = DataExtractor(hh_filepath, HH_COLS)
    trans = DataExtractor(trans_filepath, TRANS_COLS)
    dem = DataExtractor(dem_filepath, DEM_COLS)

    # read csv using dask
    dask_hh = hh.csv_reader()
    dask_trans = trans.csv_reader()
    dask_dem = dem.csv_reader()

    # convert dask df to pandas df
    pan_hh = hh.dataframe_converter(dask_hh)
    pan_trans = trans.dataframe_converter(dask_trans)
    pan_dem = dem.dataframe_converter(dask_dem)

    # DataProcessor initialization
    dataProcessorObj = DataProcessor()

    # convert all query times to datetime stamp
    start_date = dataProcessorObj.string_to_timestamp_converter(CAMPAIGN_START_DATE)
    end_date = dataProcessorObj.string_to_timestamp_converter(CAMPAIGN_END_DATE)

    # do immer join
    inner = dataProcessorObj.dataframe_merger(pan_trans, pan_hh, COMMON_KEY_INNER_MERGE, ORIENTATION)

    # SQLquery initialization
    sqlQueryObj = SQLQuery()

    # feature2
    f2_obj = sqlQueryObj.feature2(inner, DATE_COLUMN_NAME, start_date)

    # feature3
    f3_obj = sqlQueryObj.feature3(inner, DATE_COLUMN_NAME, start_date, end_date)

    # feature4
    f4_obj = sqlQueryObj.feature4(inner, COMMON_KEY_GROUP_BY, COUNT_COL_NAME)

    # feature1
    dframe_list = [pan_dem, f2_obj, f3_obj, f4_obj]
    final_table = sqlQueryObj.feature1(dframe_list, COMMON_KEY_GROUP_BY)

    # Loader initialization
    loaderObj = Loader()

    # write to csv output
    loaderObj.write_csv(final_table, csv_filepath)


if __name__ == "__main__":
    main()
