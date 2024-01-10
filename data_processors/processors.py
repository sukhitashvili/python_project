import os
from pathlib import Path
from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
from tqdm import tqdm


class CSVProcessor:
    def __init__(self, csv_path: Union[str, Path]):
        self.data = pd.read_csv(csv_path)
        self.y_cols = self.data.columns[1:]
        self.figsize = (13, 5)  # let it be statically typed and use for every visualization

    def __getitem__(self, item):
        return self.data[item]

    def save_to_sql(self, file_path, suffix, rename_columns: Optional[dict] = None, data: Optional[DataFrame] = None,
                    *args, **kwargs) -> DataFrame:
        """
        Saves dataframes as a SQLite database
        Args:
            file_path: where to save
            suffix: what name to add as suffix
            rename_columns: dict with key of existing col names and values of new column name to rename to
            data: external dataframe to safe (if provided)

        Returns:
            saved dataframe
        """
        os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)
        engine = create_engine(f'sqlite:///{file_path}.db', echo=False)
        cp_dataframe = data if (data is not None) else self.data.copy(deep=True)
        if rename_columns:
            cp_dataframe.rename(columns=rename_columns, inplace=True)
        cp_dataframe.columns = [name.capitalize() + suffix for name in cp_dataframe.columns]
        cp_dataframe.set_index(cp_dataframe.columns[0], inplace=True)
        cp_dataframe.to_sql(
            file_path,
            engine,
            if_exists="replace",
            index=True,
            *args,
            **kwargs
        )
        return cp_dataframe

    def plot_raw_data(self, plot_title: str, save_path: str = '', cols_to_plot: list = [],
                      sort_by: Optional[str] = None):
        """
        Plots raw data
        Args:
            plot_title: title of the plot
            save_path: path where to save data, if empty then figure will be shown
            cols_to_plot: columns names to plot

        Returns:
            None
        """
        plt.figure(figsize=self.figsize)
        plt.title(plot_title)
        col_names = self.y_cols if len(cols_to_plot) == 0 else cols_to_plot
        # sort values if needed
        data = None
        if sort_by:
            data = self.data.sort_values(by=sort_by)
        else:
            data = self.data
        # plot data
        for y_col in col_names:
            plt.plot(data['x'], data[y_col], label=y_col.title())
        # add legend and x axis label
        plt.legend()
        plt.xlabel('X Values')
        self.save_or_show(save_path=save_path)

    @staticmethod
    def save_or_show(save_path: str = ''):
        if save_path:
            folder = os.path.dirname(save_path)
            os.makedirs(folder, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_multiple_dataframes(self, list_of_dfs: List[DataFrame], cols_to_plot: List[List[str]],
                                 plot_title: str, save_path: str = ''):
        plt.figure(figsize=self.figsize)
        plt.title(plot_title)

        for curr_df, df_cols in zip(list_of_dfs, cols_to_plot):
            for y_col in df_cols:
                plt.plot(curr_df['x'], curr_df[y_col], label=y_col.title())

        plt.legend()
        plt.xlabel('X Values')
        self.save_or_show(save_path=save_path)


class TrainCSVProcessor(CSVProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_cols_to_ideals_ = dict()  # will be created by self.match_to_ideals function
        self.max_diffs_dict = dict()  # will be created by find_max_differences_to_mappings function

    def match_to_ideals(self, ideals_df: DataFrame) -> Dict:
        """
        Matches Y columns from ideals csv to Ys in train csv
        Args:
            ideals_df: ideals csv dataframe

        Returns:
            (dict): dict keys are train col names and values are corresponding matched Ys from ideal csv
        """
        dropped_ideal_csv = ideals_df.drop(columns='x')
        train_cols_to_ideals = {}
        for trn_col_name in tqdm(self.y_cols):
            tr_col = self[trn_col_name]
            diffs = (dropped_ideal_csv.sub(tr_col, axis=0) ** 2)
            diffs_sum = diffs.sum(axis=0)
            closest_index = diffs_sum.argmin() + 1  # +1 as we dropped the first colum, so in original df it will be next column
            train_cols_to_ideals[trn_col_name] = f'y{closest_index}'

        self.train_cols_to_ideals = train_cols_to_ideals
        return self.train_cols_to_ideals

    def vis_mapping(self, ideals_df: DataFrame, save_path: str = ''):
        """
        Visualizes subset of columns from ideals csv dataframe which were matched to train columns
        Args:
            ideals_df: ideals csv dataframe
            save_path:

        Returns:
            None
        """
        plt.figure(figsize=self.figsize)
        plt.title('Ideals to train')
        for tr_col in self.y_cols:
            plt.plot(self['x'], self[tr_col], label=f'Train {tr_col}')
            plt.plot(ideals_df['x'], ideals_df[self.train_cols_to_ideals[tr_col]],
                     label=f'Ideal {self.train_cols_to_ideals[tr_col]}')
        legend = plt.legend(prop={'size': 10})
        self.save_or_show(save_path=save_path)

    def filter_ideals(self, ideals_processor: CSVProcessor) -> DataFrame:
        """
        Returns x and Ys which are mapped to the train cols
        Args:
            ideals_processor: ideals csv dataframe

        Returns: None
        """
        mapped_y_cols = list(self.train_cols_to_ideals.values())
        estimated_ideals = ideals_processor[['x'] + mapped_y_cols]
        return estimated_ideals

    def find_max_differences_to_mappings(self, ideals_processor: CSVProcessor) -> Dict:
        estimated_ideals = self.filter_ideals(ideals_processor=ideals_processor)
        renamed_ideals = estimated_ideals.rename(
            columns={ideal_y: train_y for train_y, ideal_y in self.train_cols_to_ideals.items()})
        # check is names are matched
        assert all(renamed_ideals.columns == self.data.columns), \
            "Columns do not match, probably moved or renamed incorrectly!"

        diffs_df = (self.data - renamed_ideals).rename(columns=self.train_cols_to_ideals)
        abs_diffs_scaled = abs(diffs_df).max(axis=0) * np.sqrt(2)
        max_diffs = dict(abs_diffs_scaled)
        del max_diffs['x']
        self.max_diffs_dict = max_diffs
        return max_diffs


class IdealCSVProcessor(CSVProcessor):
    pass


class TestCSVProcessor(CSVProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_test_df = None  # fill be created by self.assign_to_ideals

    def save_to_sql(self, data: Optional[DataFrame] = None, *args, **kwargs) -> DataFrame:
        if data is None:
            if self.assigned_test_df is None:
                raise RuntimeError(
                    "self.assigned_test_df is None, possibly you missed to run self.assign_to_ideals function before"
                    " self.save_to_sql!")
            data = self.assigned_test_df.copy(deep=True)
        return super().save_to_sql(data=data, *args, **kwargs)

    def assign_to_ideals(self, train_processor: TrainCSVProcessor, ideals_processor: IdealCSVProcessor) -> DataFrame:
        """
        Assigns x,y point pairs from the test csv to the subset of ideals which were matched with columns of train.
        So x,y point from test will be assigned to one of the 4 columns from matched ideals.
        Args:
            train_processor:  train processor class which acts like a dataframe
            ideals_processor: ideals processor class which acts like a dataframe

        Returns:
            test dataframe which contains all the information about assignment:
                - ideal column index and corresponding value of the ideal point
                - delta deviation
        """
        estimated_ideals = train_processor.filter_ideals(ideals_processor=ideals_processor)
        # validate='many_to_one' is used as test.csv has duplicated entries in X column
        merged_ideals_test = pd.merge(self.data, estimated_ideals, on='x', how='left', validate='many_to_one').rename(
            columns={'y': 'test'})
        assert sum(self['x'].values != merged_ideals_test['x'].values) == 0, "X column values do not match!"

        # find the absolute difference
        mapped_ideals_cols = list(train_processor.train_cols_to_ideals.values())
        subset_ideals = merged_ideals_test[mapped_ideals_cols]
        ideals_test_abs_diff = abs(subset_ideals.sub(merged_ideals_test['test'], axis=0))

        # assign test x,y pairs to mapped ideals
        mapped_ideal_cols = list(train_processor.train_cols_to_ideals.values())
        max_diffs = train_processor.find_max_differences_to_mappings(ideals_processor=ideals_processor)

        merged_ideals_test['No. of ideal func'] = ideals_test_abs_diff.apply(
            lambda row: self.find_closest(row, col_names=mapped_ideal_cols, max_diffs=max_diffs), axis=1)
        merged_ideals_test['closest_ideal_values'] = merged_ideals_test.apply(
            lambda row: self.select_value(row, col_name='No. of ideal func'), axis=1)
        merged_ideals_test["Delta Y (test func)"] = abs(
            merged_ideals_test['test'] - merged_ideals_test['closest_ideal_values'])

        self.assigned_test_df = merged_ideals_test
        return self.assigned_test_df

    def plot_with_assigned(self, sort_by: str = 'x', save_path: str = ''):
        sorted_df = self.assigned_test_df  # just call it sorted even if it will not be sorted
        if sort_by:
            sorted_df = self.assigned_test_df.sort_values(by=[sort_by])

        # plot test x,y
        test_y = sorted_df['test'].values
        x_values = sorted_df['x'].values
        plt.figure(figsize=self.figsize)
        plt.title('Graph of values from test.csv and closes ideals')
        plt.plot(x_values, test_y, label='Test')

        #  now plot columns of ideals which were assigned after dropping nans and annotate scatter plot with col names
        without_nans = sorted_df.dropna()
        closes_ideals_names = without_nans['No. of ideal func'].values
        closes_ideals_values = without_nans['closest_ideal_values'].values
        x_values = without_nans['x'].values

        # plot and annotate
        plt.scatter(x_values, closes_ideals_values, label='Closes ideals', color='red')
        # annotate ideals with column names
        for i, (txt, x, y) in enumerate(zip(closes_ideals_names, x_values, closes_ideals_values)):
            # tweak x,y coords of the text so that title of nearby points will not overlap
            y = y + 3 if (i % 2 == 0) else y - 3
            x = x + 0.2 if (i % 2 == 0) else x - 0.2
            plt.annotate(txt, (x, y), size=9)

        legend = plt.legend(prop={'size': 10})
        plt.xlabel('X Values')
        self.save_or_show(save_path=save_path)

    @staticmethod
    def find_closest(row, col_names: list, max_diffs: dict) -> str:
        """
        Processes data-frame's rows.
        Finds the closest column in a given row which must be closer
        than maximum difference of that column estimated by a criterion from the task description.
        Args:
            row: dataframe row
            col_names: which cols to look for
            max_diffs: max difference needed for classification criterion

        Returns:
            column name Y_n or '-' if value is more than maximum difference
        """
        min_val = float('inf')
        closest_col = None
        for col in col_names:
            gt_value = row[col]
            if gt_value < min_val:
                min_val = gt_value
                closest_col = col
        if min_val <= max_diffs[closest_col]:
            return closest_col
        return '-'

    @staticmethod
    def select_value(row, col_name: str = ''):
        """
        Processes data-frame's rows.
        returns Y_n's corresponding value of the closest ideal column or nan if it has not matched
        Args:
            row: dataframe row
            col_name: col name which contains columns' names in rows

        Returns: value or np.nan
        """
        if row[col_name] == '-':
            return np.nan
        return row[row[col_name]]
