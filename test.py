import gym

env = gym.make('gym_ball:ball-v0')
env.reset()
while True:
    env.step(150)
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
