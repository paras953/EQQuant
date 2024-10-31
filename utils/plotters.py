import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple
from matplotlib.backends.backend_pdf import PdfPages


def line_plot(df: pd.DataFrame, column_list: List[str], title: str, data_type: str,
              figsize: Tuple[int, int]) -> plt.figure:
    """
    :param df: pandas df having datetime index
    :param column_list:  list of columns you want to plot data for
    :param title: title of the plot
    :param data_type: type of data you are plotting
    :param figsize: size of the plot
    :return: a line plot fig
    """
    fig, ax = plt.subplots(figsize=figsize)
    for cols in column_list:
        ax.plot(df.index, df[cols], linestyle='-', label=cols, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel('Period')
    ax.set_ylabel(data_type)
    ax.grid(visible=True, which='major', axis='y', alpha=0.2, color='r')
    ax.legend()
    plt.close(fig)
    return fig

def combine_plot(fig_list:List[plt.figure],output_path:str,file_name:str) -> None:
    """
    Saves the plots as a pdf in the given directory
    :param fig_list: a list of matplotlib figs (i.e., plots)
    :param output_path: directory where pdf will be saved
    :param file_name: name of the file without any extension
    :return: None
    """
    if not os.path.exists(output_path): os.makedirs(output_path)
    file_name = output_path +'/' + file_name +'.pdf'
    with PdfPages(file_name) as doc:
        for fig in fig_list:
            fig.savefig(doc,format='pdf')

    print(f'File saved as -> {file_name}')
    return None

def multi_bar_plot(df:pd.DataFrame,x_column:str,y_column_list:List[str],data_type:str,title:str)->plt.figure:
    """

    :param df: a pandas df
    :param x_column: x-axis column name
    :param y_column_list: y-axis column name
    :param data_type : type of data that is being plotted
    :param title : title of the plot
    :return: figure object
    """
    # Set up dynamic bar width and positions based on the number of categories
    num_categories = len(y_column_list)
    bar_width = 0.8 / num_categories  # Width is adjusted to fit all categories
    x = np.arange(len(df))  # Position of each group on the x-axis

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, cat_type in enumerate(y_column_list):
        ax.bar(x + i * bar_width - (bar_width * (num_categories - 1) / 2),
               df[cat_type], width=bar_width, label=cat_type)

    ax.set_xlabel(x_column)
    ax.set_ylabel(data_type)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_column],rotation=90)
    ax.legend(title="Period")
    ax.grid(visible=True, which='major', axis='y', alpha=0.2, color='r')
    ax.set_title(title)
    plt.close(fig)
    return fig
