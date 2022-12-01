'''
实验前提：采集数据时手机x轴与重力方向垂直

1.Model
参数列表（3个参数）：
线性加速度矩阵（x轴加速度、y轴加速度、z轴加速度）；
重力加速度矩阵（x轴重力加速度、y轴重力加速度、z轴重力加速度）;
四元数矩阵（四元数x、四元数y、四元数z、四元数w）

2.Model参数类型：
numpy.ndarray
'''

import math
import numpy as np
from scipy.stats import norm
from scipy import interpolate
from math import degrees, radians, fmod, atan2

class Model(object):
    def __init__(self, li, gr, ma,input,time):
        self.linear = li
        self.gravity = gr
        self.magnetometer = ma
        self.input = input  #0:维度，1：经度，2：方向
        self.time = time

    '''
        用accelerometer 和 magnetometer计算欧拉角
    '''
    def obtain_euler(self):
        #TODO:check, accelerometer 用带重力的，以后试试不带重力的
        accelX = self.gravity[:,0]
        accelY = self.gravity[:,1]
        accelZ = self.gravity[:,2]
        magX = self.magnetometer[:,0]
        magY = self.magnetometer[:,1]
        magZ = self.magnetometer[:,2]

        pitch = np.array([atan2(accelY[i], np.sqrt(accelX[i] **2 + accelZ[i]**2)) \
            for i in range(len(accelX))])
        roll = np.array([atan2(-accelX[i], np.sqrt(accelY[i]**2 + accelZ[i]**2)) \
            for i in range(len(accelX))])
        
        Yh = (magY * np.cos(roll)) - (magZ * np.sin(roll))
        Xh = (magX * np.cos(pitch))+(magY * np.sin(roll)*np.sin(pitch)) + (magZ * np.cos(roll) * np.sin(pitch))
        yaw = np.array([atan2(Yh[i],Xh[i]) for i in range(len(magX))])
        return pitch, roll, yaw
    
    '''
        获得手机坐标系与地球坐标系之间的角度（theta）
    '''
    def coordinate_conversion(self):
        gravity = self.gravity
        linear = self.linear

        # g_x = gravity[:, 0]
        g_y = gravity[:, 1]
        g_z = gravity[:, 2]

        # linear_x = linear[:, 0]
        linear_y = linear[:, 1]
        linear_z = linear[:, 2]
        
        theta = np.arctan(np.abs(g_z/g_y))

        # 得到垂直方向加速度（除去g）
        a_vertical = linear_y*np.cos(theta) + linear_z*np.sin(theta)

        return a_vertical
    
    '''
        步数检测函数

        walkType取值：
        normal：正常行走模式
        abnormal：融合定位行走模式（每一步行走间隔大于1s）

        返回值：
        steps
        字典型数组，每个字典保存了峰值位置（index）与该点的合加速度值（acceleration）
    '''
    def step_counter(self, frequency=100, walkType='normal'):
        offset = frequency/100
        g = 9.794
        a_vertical = self.coordinate_conversion()
        slide = 40 * offset # 滑动窗口（100Hz的采样数据）
        frequency = 100 * offset
        # 行人加速度阈值
        min_acceleration = 0.2 * g # 0.2g
        max_acceleration = 2 * g   # 2g
        # 峰值间隔(s)
        # min_interval = 0.4
        min_interval = 0.4 if walkType=='normal' else 3 # 'abnormal
        # max_interval = 1
        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': 0}

        # 以40*offset为滑动窗检测峰值
        # 条件1：峰值在0.2g~2g之间
        for i, v in enumerate(a_vertical):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i%slide == 0 and peak['index'] != 0:
                steps.append(peak)
                peak = {'index': 0, 'acceleration': 0}
        
        # 条件2：两个峰值之前间隔至少大于0.4s*offset
        # del使用的时候，一般采用先记录再删除的原则
        if len(steps)>0:
            lastStep = steps[0]
            dirty_points = []
            for key, step_dict in enumerate(steps):
                # print(step_dict['index'])
                if key == 0:
                    continue
                if step_dict['index']-lastStep['index'] < min_interval*frequency:
                    # print('last:', lastStep['index'], 'this:', step_dict['index'])
                    if step_dict['acceleration'] <= lastStep['acceleration']:
                        dirty_points.append(key)
                    else:
                        lastStep = step_dict
                        dirty_points.append(key-1)
                else:
                    lastStep = step_dict
            
            counter = 0 # 记录删除数量，作为偏差值
            for key in dirty_points:
                del steps[key-counter]
                counter = counter + 1
        
        return steps
    
    # 步长推算
    # 目前的方法不具备科学性，临时使用
    def step_stride(self, max_acceleration):
        return np.power(max_acceleration, 1/4) * 0.5

    # 航向角
    # 根据姿势直接使用yaw
    def step_heading(self):
        _, _, yaw = self.obtain_euler()
        # init_theta = yaw[0] # 初始角度
        for i,v in enumerate(yaw):
            # yaw[i] = -(v-init_theta)
            yaw[i] = -v # 由于yaw逆时针为正向，转化为顺时针为正向更符合常规的思维方式
        return yaw
    
    '''
        步行轨迹的每一个相对坐标位置, offset表示初始角度
        返回的是预测作为坐标
    '''
    def pdr_position(self, frequency=100, walkType='normal', offset=0, initPosition=(0, 0)):
        #TODO check! 这里返回弧度制
        yaw = self.step_heading()
        steps = self.step_counter(frequency=frequency, walkType=walkType)
        position_x = []
        position_y = []
        x = initPosition[0]
        y = initPosition[1]
        position_x.append(x)
        position_y.append(y)
        strides = []
        angle = [offset]
        time = [0]
        for v in steps:
            index = v['index']
            length = self.step_stride(v['acceleration'])
            strides.append(length)
            theta = yaw[index] + radians(offset)
            angle.append(degrees(theta))
            x = x + length*np.sin(theta)
            y = y + length*np.cos(theta)
            position_x.append(x)
            position_y.append(y)
            time.append(self.time[index])
        # 步长计入一个状态中，最后一个位置没有下一步，因此步长记为0
        return position_x, position_y, strides + [0], angle,time

    '''
        对于输入data，用10%时间的位置解（相对位置，角）与（经纬度，方向）关系，用来算后面的位置与经纬度关系
    '''
    def predict_position(self,start_index,predic_time:np.ndarray):
        #start_index 表示从time中的哪个index 开始预测，
        #TODO：check!这里我们是去掉表头了！故比写进csv的行号少1！
        noise = 0.2
        x_offset = self.input[:,0][start_index-2]*(1-noise) + self.input[:,0][start_index-3]*noise
        y_offset = self.input[:,1][start_index-2]*(1-noise) + self.input[:,1][start_index-3]*noise
        n_x,n_y,_,angle,step_time=self.pdr_position()    #TODO check!!!突然想起来算的angle是弧度制啊！！

        #interpolate
        tck_x = interpolate.splrep(step_time,np.array(n_x))
        tck_y = interpolate.splrep(step_time,np.array(n_y))
        tck_a = interpolate.splrep(step_time,np.array(angle))

        tmp_time = np.array(predic_time,dtype=np.float64)
        move_x = interpolate.splev(tmp_time,tck_x)
        move_y = interpolate.splev(tmp_time,tck_y)
        move_a = interpolate.splev(tmp_time,tck_a)

        ans_x = (np.array(move_x) * 0.000009) + x_offset
        ans_y = (np.array(move_y) * 0.00001141) + y_offset
        ans_a = np.array([fmod(i+720,360) for i in move_a]) #我决定插值之后再mod

        return ans_x,ans_y,ans_a
        


