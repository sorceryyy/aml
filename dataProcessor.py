import pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataProcessor():
    def __init__(self) -> None:
        '''attention: all data should name in right format!'''
        self.path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
        self.acc_file = os.path.join(self.path,"Accelerometer.csv")
        self.bar_file = os.path.join(self.path,"Barometer.csv")    #加速度计
        self.gyr_file = os.path.join(self.path,"Gyroscope.csv")    #气压计
        self.lin_file = os.path.join(self.path,"Linear Accelerometer.csv") #TODO:check whether blank ok!陀螺仪
        self.mag_file = os.path.join(self.path,"Magnetometer.csv")   #磁力
        self.inp_file = os.path.join(self.path,"Location_input.csv")   #位置输入location_input

        self.data = {"acc":pd.read_csv(self.acc_file,header=None), \
                    "bar": pd.read_csv(self.bar_file,header=None),\
                    "gyr": pd.read_csv(self.gyr_file,header=None),\
                    "lin": pd.read_csv(self.lin_file,header=None),\
                    "mag": pd.read_csv(self.mag_file,header=None),\
                    "inp": pd.read_csv(self.inp_file,header=None),\
                    }
        #for pdr
        linear =(self.data["lin"].values)[1:,1:].astype(np.float64)
        gravity = (self.data["acc"].values)[1:,1:].astype(np.float64)
        magnetometer = (self.data["mag"].values)[1:,1:].astype(np.float64)
        location_input = (self.data["inp"].values)[1:,[1,2,5]].astype(np.float64) #TODO: write in more stable way
        time = (self.data["mag"].values)[1:,0].astype(np.float64)

        self.pdr_model = pdr.Model(li=linear,gr=gravity,ma=magnetometer,pre_locate=location_input,time=time)

        #计算开始是NAN的行数
        self.start_nan = len(self.data["inp"]) - len(self.data["inp"].dropna(axis=0,how="any"))

    def get_pdr(self)->pdr.Model:
        '''return the pdr_model'''
        return self.pdr_model

    def save_csv(file:str):
        '''input the data, save as csv'''
        pass

