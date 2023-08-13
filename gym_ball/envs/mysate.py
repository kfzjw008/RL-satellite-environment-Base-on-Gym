import gym
import numpy as np
from gym import spaces
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 定义动作空间
action_space = spaces.Dict({
    '相机开关机控制': spaces.Discrete(2),  # 关机和开机
    '航天器充电控制': spaces.Discrete(2),  # 不充电和充电
    '航天器数传控制': spaces.Discrete(2),  # 不数传和数传
    '航天器期望飞行速度': spaces.Box(low=0, high=30, shape=(1,), dtype=float),  # 取值范围为0-30的实数
    '航天器飞行角度': spaces.Box(low=0, high=360, shape=(1,), dtype=float),  # 取值范围为0-360的实数
    '航天器探测目标点x轴': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=float),  # 取值范围为-1000-1000的实数
    '航天器探测目标点y轴': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=float),  # 取值范围为-1000-1000的实数
})


# 定义观测空间
observation_space = spaces.Dict({
    '航天器所处X坐标': spaces.Box(low=0, high=1000, shape=(1,), dtype=float),
    '航天器所处Y坐标': spaces.Box(low=0, high=1000, shape=(1,), dtype=float),
    '发现的目标所处X坐标': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=float),
    '发现的目标所处Y坐标': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=float),
    '航天器剩余电量': spaces.Box(low=0, high=100, shape=(1,), dtype=float),
    '航天器剩余存储空间': spaces.Box(low=0, high=1024, shape=(1,), dtype=float),
    '航天器剩余燃油量': spaces.Box(low=0, high=1000, shape=(1,), dtype=float),
    '相机当前状态': spaces.Discrete(2),  # 关机和开机
    '目标观测价值': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
    '航天器朝向角度': spaces.Box(low=0, high=360, shape=(1,), dtype=float),
    '航天器线速度': spaces.Box(low=0, high=30, shape=(1,), dtype=float),
    '目标朝向角度': spaces.Box(low=0, high=360, shape=(1,), dtype=float),
    '目标线速度': spaces.Box(low=0, high=10, shape=(1,), dtype=float),
    '当前数据平均价值': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
    '当前数传效率': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
    '相机开机时间周期': spaces.Box(low=0, high=100, shape=(1,), dtype=float),
    '当前位置是否可充电': spaces.Discrete(2)  # 是和否
})





# 创建自定义环境
class BallEnv(gym.Env):
    def __init__(self):
        # 定义动作空间和观测空间
        self.action_space = action_space
        self.observation_space = observation_space #状态空间和观测空间定义

        # Create a figure and axis for visualization
        self.fig, self.ax = plt.subplots()

        # Set axis limits
        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(-1000, 1000)

        # Create patches for Earth, spacecraft, and target
        self.earth_patch = patches.Circle((0, 0), radius=1000, facecolor='navy')
        self.spacecraft_patch = patches.Circle((0, 0), radius=10, facecolor='red')
        self.target_patch = patches.Circle((100, 100), radius=5, facecolor='black')

        # Add patches to the axis
        self.ax.add_patch(self.earth_patch)
        self.ax.add_patch(self.spacecraft_patch)
        self.ax.add_patch(self.target_patch)

    def step(self, action):
        self.update()

        # Return the next observation, reward, done flag, and additional info
        return observation, reward, done, info
        # 执行动作并返回新的状态、奖励、是否终止等信息
        ...

    def reset(self):
        # 重置环境并返回初始观测
        ...

    def render(self):
        # 可选的渲染方法
        ...
