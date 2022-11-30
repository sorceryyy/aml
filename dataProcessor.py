import pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataProcessor():
    def __init__(self) -> None:
        '''attention: all data should name in right format!'''
        self.path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
        self.acc_file = self.path + "Accelerometer.csv"
        self.bar_file = self.path + "Barometer.csv"    #加速度计
        self.gyr_file = self.path + "Gyroscope.csv"    #气压计
        self.lin_file = self.path + "Linear Accelerometer.csv" #TODO:check whether blank ok!陀螺仪
        self.mag_file = self.path + "Magnetometer.csv"   #磁力
        self.inp_file = self.path + "Location_input.csv"   #位置输入location_input

        self.data = {"acc":pd.read_csv(self.acc_file), \
                    "bar": pd.read_csv(self.bar_file),\
                    "gyr": pd.read_csv(self.gyr_file),\
                    "lin": pd.read_csv(self.lin_file),\
                    "mag": pd.read_csv(self.mag_file),\
                    "inp": pd.read_csv(self.inp_file),\
                    }
        #for pdr
        linear = self.data["lin"].columns[1:4].values
        self.pdr_data

