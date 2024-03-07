import os
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


class CSVProcessor:
    def __init__(self, csv_path: Union[str, Path]):
        self.data = pd.read_csv(csv_path)
        self.y_cols = self.data.columns[1:]
        self.figsize = (9, 5)  # let it be statically typed and use for every visualization

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
            *args, **kwargs: passed to DataFrame.to_sql
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
                      sort_by: Optional[str] = None, plot_type: str = 'plot'):
        """
        Plots raw data
        Args:
            plot_title: title of the plot
            save_path: path where to save data, if empty then figure will be shown
            cols_to_plot: columns names to plot
            sort_by: which column values to use for sorting
            plot_type: string representing what pyplot's function to use
                       Default to plt.plot

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
            plot_func = getattr(plt, plot_type)
            plot_func(data['x'], data[y_col], label=y_col.title())
        # add legend and x axis label
        # plt.legend()
        # plt.xlabel('X Values')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
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
                                 plot_types: List[str], plot_title: str, save_path: str = ''):
        plt.figure(figsize=self.figsize)
        plt.title(plot_title)

        for curr_df, df_cols, plt_type in zip(list_of_dfs, cols_to_plot, plot_types):
            for y_col in df_cols:
                plot_func = getattr(plt, plt_type)
                plot_func(curr_df['x'], curr_df[y_col], label=y_col.title())

        plt.legend()
        # plt.xlabel('X Values')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        self.save_or_show(save_path=save_path)
