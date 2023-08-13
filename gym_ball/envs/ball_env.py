import math
import random
import time

import gym
import numpy as np
from gym import spaces
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym.envs.classic_control import rendering
#速度银子
YZ=5
#允许运动痕迹
ALLOW_LINE=False
# 视窗大小
VIEWPORT_W = 1500
VIEWPORT_H = 900
BALL_TYPE_SELF = 1
BALL_START_ID = 0
# 球的数量
MAX_BALL_NUM = 1
MAX_BALL_SCORE = 200
#缩放因子
SCALE =1
# 卫星大小
SATELLITE_RADIUS = 10

#数传区域
DATA_CIRCLE=100


#拍照区域
DATA_CIRCLE2=100

def calculate_distance(x1, y1, x2, y2):

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


# 定义动作空间
action_space = spaces.Dict({
    '相机开关机控制': spaces.Discrete(2),  # 关机和开机
    '航天器充电控制': spaces.Discrete(2),  # 不充电和充电
    '航天器数传控制': spaces.Discrete(2),  # 不数传和数传
    '航天器期望飞行速度': spaces.Box(low=0, high=30, shape=(1,), dtype=float),  # 取值范围为0-30的实数
    '航天器飞行角度': spaces.Box(low=0, high=360, shape=(1,), dtype=float),  # 取值范围为0-360的实数
    '航天器探测目标点x轴': spaces.Box(low=0, high=VIEWPORT_W, shape=(1,), dtype=float),  # 取值范围为-1000-1000的实数
    '航天器探测目标点y轴': spaces.Box(low=0, high=VIEWPORT_H, shape=(1,), dtype=float),  # 取值范围为-1000-1000的实数
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

observation = {
    '航天器所处X坐标': 0,
    '航天器所处Y坐标': 0,
    '发现的目标所处X坐标': 0,
    '发现的目标所处Y坐标': 0,
    '航天器剩余电量': 0,
    '航天器剩余存储空间': 0,
    '航天器剩余燃油量': 0,
    '相机当前状态': 0,  # 关机和开机
    '目标观测价值': 0,
    '航天器朝向角度': 0,
    '航天器线速度': 0,
    '目标朝向角度': 0,
    '目标线速度': 0,
    '当前数据平均价值': 0,
    '当前数传效率': 0,
    '相机开机时间周期': 0,
    '当前位置是否可充电': 0  # 是和否
}

# 创建航天器
class htq():
    def __init__(self, x: np.float32, y: np.float32):
        '''
            x   coordinate
            y   coordinate
            dl  电量
                   self.store=1024 存储
        self.oil=1000 燃料
        self.camera=False 相机开关
                self.chargeC=False 充电开关
        self.dataC=False 存储开关
        self.angle=0  角度
        self.speed=0 线速度
        self.storevalue=0 存储平均价值
        self.datatransvalue=0 数据传输价值
        self.cameratime=0 相机时间
        self.allowcharge=False 允许充电

        '''
        self.x = x
        self.y = y
        self.w=0
        self.dl = 100
        self.store = 1024
        self.oil = 1000
        self.camera = False
        self.chargeC = False
        self.dataC = False
        self.angle = 0
        self.speed = 0.5*YZ
        self.storevalue = 0
        self.datatransvalue = 0
        self.cameratime = 0
        self.allowcharge = False
        self.lastupdate = time.time()  # last update time, used to caculate ball move
        self.timescale = 100  # time scale, used to caculate ball move
        self.w = 1 * 2 * math.pi / 360.0  # angle to radius
        self.t = 1
        self.ww=random.uniform(-0.01, 0.01)

    def update(self, way):
        '''
            update ball, include position
        '''
        # can only change self way
        if self.t == BALL_TYPE_SELF:
            angle = way * 2 * math.pi / 360.0  # angle to radius
            # angle += random.uniform(-1, 1)

            self.w += self.ww

        speed = self.speed  # score to speed
        now = time.time()  # now time

        delta_time = now - self.lastupdate

        # Calculate new position
        if self.t == BALL_TYPE_SELF:
            direction = np.array([math.cos(self.w), math.sin(self.w)])  # current direction
            rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            new_direction = np.dot(rotation_matrix, direction)  # apply rotation
            self.x += new_direction[0] * speed * delta_time * self.timescale
            self.y += new_direction[1] * speed * delta_time * self.timescale
        else:
            self.x += math.cos(self.w) * speed * delta_time * self.timescale
            self.y += math.sin(self.w) * speed * delta_time * self.timescale

        self.x = CheckBound(0, VIEWPORT_W, self.x)
        self.y = CheckBound(0, VIEWPORT_H, self.y)
        self.lastupdate = now  # update time


def GenerateBallID():
    global BALL_START_ID

    BALL_START_ID += 1

    return BALL_START_ID


def CheckBound(low, high, value):
    if value > high:
        value -= (high - low)
    elif value < low:
        value += (high - low)
    return value


class Ball():
    def __init__(self, x: np.float32, y: np.float32, score: np.float32, way: int, t: int):
        '''
            x   coordinate
            y   coordinate
            s   score of ball
            w   move direction of ball, in radius
            t   type of ball, self or other
        '''
        self.trajectory = []
        self.x = x
        self.y = y
        self.s = CheckBound(0, MAX_BALL_SCORE, score)
        self.w = way * 2 * math.pi / 360.0  # angle to radius
        self.t = t
        self.ww=random.uniform(-0.01, 0.01)

        self.id = GenerateBallID()  # ball id
        self.lastupdate = time.time()  # last update time, used to caculate ball move
        self.timescale = 100  # time scale, used to caculate ball move

    def update(self, way):
        '''
            update ball, include position
        '''
        #self.trajectory.append((self.x, self.y))
        # can only change self way
        if self.t == BALL_TYPE_SELF:
            angle = way * 2 * math.pi / 360.0  # angle to radius
            #angle += random.uniform(-1, 1)

            self.w += self.ww

        speed = 5.0 / self.s*YZ  # score to speed
        now = time.time()  # now time

        delta_time = now - self.lastupdate

        # Calculate new position
        if self.t == BALL_TYPE_SELF:
            direction = np.array([math.cos(self.w), math.sin(self.w)])  # current direction
            rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            new_direction = np.dot(rotation_matrix, direction)  # apply rotation
            self.x += new_direction[0] * speed * delta_time * self.timescale
            self.y += new_direction[1] * speed * delta_time * self.timescale
        else:
            self.x += math.cos(self.w) * speed * delta_time * self.timescale
            self.y += math.sin(self.w) * speed * delta_time * self.timescale

        self.x = CheckBound(0, VIEWPORT_W, self.x)
        self.y = CheckBound(0, VIEWPORT_H, self.y)
        self.lastupdate = now  # update time

    def addscore(self, score: np.float32):
        self.s += score

    def state(self):
        return [self.x, self.y, self.s, self.t]


# 创建自定义环境
def draw_satellite(center, radius):
    # 绘制卫星的主体部分（圆形）
    satellite_body = rendering.make_circle(radius)

    # 将主体部分移动到正确的位置
    satellite_body.add_attr(rendering.Transform(translation=(center[0], center[1])))

    # 设置主体部分的颜色
    satellite_body.set_color(0, 0, 255)

    # 绘制卫星的天线部分（线段）
    antenna_length = radius * 2  # 天线长度为半径的两倍
    antenna_start = (center[0], center[1])  # 天线起始点的坐标（与主体部分相同）
    antenna_end1 = (center[0], center[1] + antenna_length)  # 天线结束点的坐标
    antenna_end2 = (center[0], center[1] - antenna_length)  # 天线结束点的坐标
    antenna_end3 = (center[0] + antenna_length, center[1])  # 天线结束点的坐标
    antenna_end4 = (center[0] - antenna_length, center[1])  # 天线结束点的坐标
    satellite_antenna1 = rendering.Line(antenna_start, antenna_end1)
    satellite_antenna2 = rendering.Line(antenna_start, antenna_end2)
    satellite_antenna3 = rendering.Line(antenna_start, antenna_end3)
    satellite_antenna4 = rendering.Line(antenna_start, antenna_end4)

    # 设置天线部分的颜色
    # satellite_antenna.set_color(1, 0, 0)

    # 组合主体和天线部分
    satellite = rendering.Compound([satellite_body, satellite_antenna1, satellite_antenna2, satellite_antenna3,
                                    satellite_antenna4])
    return satellite


class BallEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.step_count =0  # 增加步数
        self.viewer = None  # render viewer
        self.scale = SCALE # render scale
        # 定义动作空间和观测空间
        self.action_space = action_space
        self.observation_space = observation_space  # 状态空间和观测空间定义

        # Create a figure and axis for visualization
        self.fig, self.ax = plt.subplots()



        self.state = np.zeros((MAX_BALL_NUM * 4,), dtype=np.float32)

        self.reset()

    def reset(self):
        # 创建观测目标
        self.balls = []


        self.htqs=htq(np.random.rand(1)[0] * VIEWPORT_W,np.random.rand(1)[0] * VIEWPORT_H)
        aaa=self.htqs.x
        # random gen other balls
        min = MAX_BALL_SCORE
        max = 0
        for i in range(MAX_BALL_NUM - 1):
            tmp = self.randball(0)

            if tmp.s < min:
                min = tmp.s

            if tmp.s > max:
                max = tmp.s

            self.balls.append(tmp)

        # random gen self ball
        self.selfball = self.randball(BALL_TYPE_SELF, (min + max) / 2)

        # add to ball list
        self.balls.append(self.selfball)

        # update state
        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])
        self.spacecraft = htq(np.random.rand(1)[0] * VIEWPORT_W, np.random.rand(1)[0] * VIEWPORT_H)


        observation = {
            '航天器所处X坐标': self.htqs.x,
            '航天器所处Y坐标': self.htqs.y,
            '发现的目标所处X坐标': 1,
            '发现的目标所处Y坐标': 1,
            '航天器剩余电量': 100,
            '航天器剩余存储空间': 1024,
            '航天器剩余燃油量': 1000,
            '相机当前状态': 0,  # 关机和开机
            '目标观测价值': 0.6,
            '航天器朝向角度': self.htqs.w,
            '航天器线速度': self.htqs.speed,
            '目标朝向角度': 0,
            '目标线速度': 0,
            '当前数据平均价值': 0,
            '当前数传效率': 1,
            '相机开机时间周期': 1,
            '当前位置是否可充电': 0  # 是和否
        }
        for item in self.state:
            pass
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]
            observation['发现的目标所处X坐标'] = item[0] * self.scale
            observation['发现的目标所处Y坐标'] = item[1] * self.scale
        return observation

    def step(self, action):
        camera_switch = action_space['相机开关机控制']
        spacecraft_charge = action_space['航天器充电控制']
        spacecraft_data_transmission = action_space['航天器数传控制']
        spacecraft_velocity = action_space['航天器期望飞行速度']
        spacecraft_angle = action_space['航天器飞行角度']
        target_point_x = action_space['航天器探测目标点x轴']
        target_point_y = action_space['航天器探测目标点y轴']


        self.step_count += 1  # 增加步数
        reward = 0.0
        done = False

        action = 10 * action
        self.htqs.update(action)
        # update ball
        for _, ball in enumerate(self.balls):
            ball.update(action)

        '''
            Calculate Ball Eat
            if ball A contains ball B's center, and A's score > B's score, A eats B
        '''
        _new_ball_types = []
        for _, A_ball in enumerate(self.balls):
            for _, B_ball in enumerate(self.balls):

                if A_ball.id == B_ball.id:
                    continue

                # radius of ball A
                A_radius = math.sqrt(A_ball.s / math.pi)

                # vector AB
                AB_x = math.fabs(A_ball.x - B_ball.x)
                AB_y = math.fabs(A_ball.y - B_ball.y)

                # B is out of A
                if AB_x > A_radius or AB_y > A_radius:
                    continue

                # B is out of A
                if AB_x * AB_x + AB_y * AB_y > A_radius * A_radius:
                    continue



        # generate balls to MAX_BALL_NUM
        for _, val in enumerate(_new_ball_types):
            self.balls.append(self.randball(int(val)))

        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])

        return self.state.reshape(MAX_BALL_NUM * 4, ), reward, done, {}



    def update(self):
        pass
        # Update the positions of spacecraft and target
        # self.spacecraft_patch.center = self.state.spacecraft_position
        # self.target_patch.center = self.state.target_position

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W * self.scale, VIEWPORT_H * self.scale)

        # add ball to viewer
        for item in self.state:
            pass
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]

            transform = rendering.Transform()
            transform.set_translation(_x, _y)

            # add a circle
            # center: (x, y)
            # radius: sqrt(score/pi)
            # colors: self in red, other in blue
            self.viewer.draw_circle(math.sqrt(_s / math.pi) * 3, 30, color=(_t, 0, 1)).add_attr(transform)
            self.viewer.draw_circle(DATA_CIRCLE2, color=(1, 1, 0), filled=False).add_attr(rendering.Transform(translation=(_x * SCALE, _y * SCALE)))

        # Draw satellite
        satellite_center = (self.htqs.x, self.htqs.y)
        satellite_radius = SATELLITE_RADIUS
        if ALLOW_LINE ==True:
            for ball in self.balls:
                for trajectory in ball.trajectory:
                    trajectory_point = rendering.make_circle(1)
                    trajectory_point.set_color(0, 0, 0)  # 设置颜色为黑色
                    trajectory_point.add_attr(
                        rendering.Transform(translation=(trajectory[0] * self.scale, trajectory[1] * self.scale)))
                    self.viewer.add_geom(trajectory_point)



        satellite_shape = draw_satellite(satellite_center, satellite_radius)
        if hasattr(self, 'satellite_shape'):
            self.viewer.geoms.remove(self.satellite_shape)
        self.satellite_shape = draw_satellite(satellite_center, satellite_radius)
        self.viewer.add_geom(self.satellite_shape)
        self.viewer.draw_circle(30, color=(0, 0, 1)).add_attr(rendering.Transform(translation=(VIEWPORT_W*0.5*SCALE,VIEWPORT_H*0.5*SCALE)))
        self.viewer.draw_circle(DATA_CIRCLE, color=(1, 1, 0), filled=False).add_attr(
            rendering.Transform(translation=(VIEWPORT_W * 0.5 * SCALE, VIEWPORT_H * 0.5 * SCALE)))
        self.viewer.draw_line((0, VIEWPORT_H*0.5*SCALE), (VIEWPORT_W, VIEWPORT_H*0.5*SCALE), color=(1, 0, 0))

        # 绘制文字标签
        fig, ax = plt.subplots(figsize=(VIEWPORT_W * self.scale, VIEWPORT_H * self.scale))
        ax.imshow(self.viewer.get_array())




        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        return

    @staticmethod
    def randball(_t: int, _s: float = 0):
        if _s <= 0:
            _s = np.random.rand(1)[0] * MAX_BALL_SCORE
        _b = Ball(np.random.rand(1)[0] * VIEWPORT_W, np.random.rand(1)[0] * VIEWPORT_H, _s,
                  int(np.random.rand(1)[0] * 360), _t)
        return _b


if __name__ == '__main__':
    env = BallEnv()

    while True:
        env.step(15)
        env.render()
