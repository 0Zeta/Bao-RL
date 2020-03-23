from random import choice

import gym
import numpy as np

env = gym.make('HusBao-v0')

nb_moves = []

for i in range(10000):
    env.reset()
    i = 0
    while not env.done:
        i += 1
        action = choice(env.get_available_actions())
        env.step(action)  # take a random action
    nb_moves.append(i)

data = np.asarray(nb_moves)
print(str(np.average(data)))
print(str(np.median(data)))
print(str(data.min()))
print(str(data.max()))

env.render()
