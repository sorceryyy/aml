from dataProcessor import DataProcessor
from math import atan2,sqrt,cos,sin,degrees,fmod

if __name__ == "__main__":
    data = DataProcessor()
    data.get_pdr_predict()
    data.save_csv("try_out.csv")

    # acY = 2.24
    # acX = 0.0467
    # acZ = 4.12
    # magX = -32.3
    # magY = -19.9
    # magZ = -30.9

    # pitch = atan2(acY,sqrt(acX**2+acZ**2) )
    # roll = atan2(-acX,sqrt(acY**2+acZ**2) )

    # Yh = (magY * cos(roll)) - (magZ*sin(roll))
    # Xh = (magX * cos(pitch)) + (magY*sin(roll)*sin(pitch)) + (magZ*cos(roll)*sin(pitch))

    # yaw = fmod(atan2(Yh,Xh)*360/3.1415926+720,360)
    # print(yaw)