import pandas as pd
from typing import *
import numpy as np
import matplotlib.pyplot as plt

class Track:
    """
    Wrapper for dataframe containing eye tracking information
    
    Attributes
    -----------
    df : pd.DataFrame
        DataFrame containing tracking information 
    raw_df : pd.DataFrame
        DataFrame containing raw tracking information for visualization
    period : int
        Number of milliseconds between observations
    img : np.ndarray
        Image on which points are tracked
    cols : Iterable[str]
        Column names for df
    """

    def __init__(self, df: pd.DataFrame, raw_df: pd.DataFrame, period: int, img: np.ndarray, cols: Iterable[str]):
        self.df = df
        self.raw_df = raw_df
        self.period = period
        self.img = img
        self.cols = cols

    @classmethod
    def from_excel(cls, file: str, period: int, img: str, time_col: str, IMG_X_col: str, IMG_Y_col: str, type_col: str, fixation_X_col: str, fixation_Y_col: str, contains_other_sensors = True) -> Self:
        """Reads tracking data from an excel spreadsheet

        Parameters
        -----------
        file : str 
            Excel file containing tracking information
        period : int
            Number of milliseconds between measurements in file 
        img : str
            Path to image file
        time_col : str
            Name of column containing timing information, must contain "Fixation" during fixations
        IMG_X_col : str
            Name of column containing X (horizontal) position in image
        IMG_Y_col : str
            Name of column containing Y (vertical) position in image
        type_col : str
            Name of column classifying eye movement types
        fixation_X_col : str
            Name of column containing Fixation X information
        fixation_Y_col : str
            Name of column containing Fixation Y information
        contains_other_sensors : bool
            Set to true if file contains sensors other than eye trackers
        Returns:
        --------
        Track
            Instance of Track
        """
        # Read file
        raw_df = pd.read_excel(file)
        
        # Remove other sensors if necessary
        if contains_other_sensors:
            raw_df = pd.DataFrame(raw_df[raw_df["Sensor"] == "Eye Tracker"])
        df = raw_df.copy()
        cols = [time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col]
        raw_df = raw_df[cols]

        # Subset to time where eye tracking starts
        mintime = df[~df[IMG_X_col].isna()][time_col].min()
        maxtime = df[~df[IMG_Y_col].isna()][time_col].max()
        df = df[cols][(df[time_col] >= mintime) & (df[time_col] <= maxtime)]

        # Filter out points with Fixations that are not found in the Image
        df = pd.DataFrame(df[~((df[type_col] == "Fixation") & df[IMG_X_col].isna())])

        # Read image
        img = plt.imread(img)

        return cls(df, raw_df, period, img, cols)