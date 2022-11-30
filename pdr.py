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
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Model(object):
    def __init__(self, li, gr, ma,pre_locate,time):
        self.linear = li
        self.gravity = gr
        self.magnetometer = ma
        self.pre_location = pre_locate
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

        pitch = np.arctan2(accelY, np.sqrt(accelX **2 + accelZ**2))
        roll = np.arctan2(-accelX, np.sqrt(accelY**2 + accelZ**2))
        
        Yh = (magY * np.cos(roll)) - (magZ * np.sin(roll))
        Xh = (magX * np.cos(pitch))+(magY * np.sin(roll)*np.sin(pitch)) + (magZ * np.cos(roll) * np.sin(pitch))
        yaw = np.arctan2(Yh,Xh)
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
        步行轨迹的每一个相对坐标位置
        返回的是预测作为坐标
    '''
    def pdr_position(self, frequency=100, walkType='normal', offset=0, initPosition=(0, 0)):
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
        time = []
        for v in steps:
            index = v['index']
            length = self.step_stride(v['acceleration'])
            strides.append(length)
            theta = yaw[index] + offset
            angle.append(theta)
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
    def predict_position(self,x_time):
        pass
        


    '''
    显示步伐检测图像
      walkType取值：
        - normal：正常行走模式
        - abnormal：融合定位行走模式（每一步行走间隔大于1s）
    '''
    def show_steps(self, frequency=100, walkType='normal'):
        a_vertical = self.coordinate_conversion()
        steps = self.step_counter(frequency=frequency, walkType=walkType)

        index_test = []
        value_test = []
        for v in steps:
            index_test.append(v['index'])
            value_test.append(v['acceleration'])

        textstr = '='.join(('steps', str(len(steps))))
        _, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.plot(a_vertical)
        plt.scatter(index_test, value_test, color='r')
        plt.xlabel('samples')
        plt.ylabel('vertical acceleration')
        plt.show()

    '''
        输出一个数据分布散点图, 用来判断某一类型数据的噪声分布情况, 通常都会是高斯分布, 
    '''
    def show_gaussian(self, data, fit):
        wipe = 150
        data = data[wipe:len(data)-wipe]
        division = 100
        acc_min = np.min(data)
        acc_max = np.max(data)
        interval = (acc_max-acc_min)/division
        counter = [0]*division
        index = []

        for k in range(division):
            index.append(acc_min+k*interval)

        for v in data:
            for k in range(division):
                if v>=(acc_min+k*interval) and v<(acc_min+(k+1)*interval):
                    counter[k] = counter[k]+1
        
        textstr = '\n'.join((
            r'$max=%.3f$' % (acc_max, ),
            r'$min=%.3f$' % (acc_min, ),
            r'$mean=%.3f$' % (np.mean(data), )))
        _, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.scatter(index, counter, label='distribution')

        if fit==True:
            length = math.ceil((acc_max-acc_min)/interval)
            counterArr = length * [0]
            for value in data:
                key = int((value - acc_min) / interval)
                if key >=0 and key <length:
                    counterArr[key] += 1
            normal_mean = np.mean(data)
            normal_sigma = np.std(data)
            normal_x = np.linspace(acc_min, acc_max, 100)
            normal_y = norm.pdf(normal_x, normal_mean, normal_sigma)
            normal_y = normal_y * np.max(counterArr) / np.max(normal_y)
            ax.plot(normal_x, normal_y, 'r-', label='fitting')

        plt.xlabel('acceleration')
        plt.ylabel('total samples')
        plt.legend()
        plt.show()


    '''
        显示PDR运动轨迹图
    '''
    def show_trace(self, frequency=100, walkType='normal', initPosition=(0, 0), **kw):
        plt.grid()
        handles = []
        labels = []

        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            l1, = plt.plot(trace_x, trace_y, color='g')
            handles.append(l1)
            labels.append('real tracks')
            plt.scatter(trace_x, trace_y, color='orange')
            for k in range(0, len(trace_x)):
                plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')

        if 'offset' in kw:
            offset = kw['offset']
        else:
            offset = 0
        
        x, y, _, _ = self.pdr_position(frequency=frequency, walkType=walkType, offset=offset, initPosition=initPosition)
        print('steps:', len(x)-1)

        for k in range(0, len(x)):
            plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        l2, = plt.plot(x, y, 'o-')
        handles.append(l2)
        labels.append('pdr predicting')
        plt.legend(handles=handles,labels=labels,loc='best')
        plt.show()