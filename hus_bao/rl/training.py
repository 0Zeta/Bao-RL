import random
from random import choice, getrandbits

import gym
import numpy as np
from sklearn.utils import shuffle

from hus_bao.rl.agents import Agent

RANDOM_STATE = 12345
random.seed(RANDOM_STATE)


def data_generator(batch_size, agent: Agent, opponents, model, gamma=0.8):
    """generates data for the model
    Arguments:
        batch_size (int): how many datapoints should be generated
        agent (Agent):    the main agent for the data
        opponents (list): the opponents for the agent
        model (Model):    the model that should be used to evaluate states
        gamma (float):    the discount factor that should be used for reward computation
    Returns:
        ndarray: the game states (after an action)
        ndarray: the computed values of the states
    """
    env = gym.make('HusBao-v0')
    while True:
        X = []
        y = []

        while len(X) < batch_size:
            env.reset()
            opponent = choice(opponents)

            if getrandbits() == 1:
                current_agent = agent  # the main agent goes first
                waiting_agent = opponent
            else:
                current_agent = opponent
                waiting_agent = agent

            states = [[], []]
            states_after_action = [[], []]
            while not env.done:
                state = np.copy(env.state)
                next_state, reward, done, _ = env.step(current_agent.move(state, env.get_available_actions()))
                states.append(state)
                states_after_action.append(next_state)
                current_agent, waiting_agent = waiting_agent, current_agent

            # Compute state values (The states are the results of actions.)
            values = [np.ndarray(shape=(len(states_after_action[0]), 1), dtype=np.float),
                      np.ndarray(shape=(len(states_after_action[1]), 1), dtype=np.float)]
            outcome = env.outcome
            values[0][-1] = outcome * 10
            values[1][-1] = -outcome * 10

            # Compute rewards
            for rewards in values:
                for i in reversed(range(len(rewards))):
                    rewards[i] += gamma * rewards[i + 1]

            # Subtract old values
            states_after_action = np.asarray(states_after_action, dtype=np.int)
            states_after_action = np.reshape(states_after_action, newshape=(2, -1, 32))
            old_estimates = [np.reshape(model.predict(states_after_action[0]), newshape=(-1, 1)),
                             np.reshape(model.predict(states_after_action[1]), newshape=(-1, 1))]
            values = [values[0] - old_estimates[0], values[1] - old_estimates[1]]

            # Add estimates of the new states

        X, y = shuffle(np.asarray(X), np.asarray(y), random_state=RANDOM_STATE)
        yield X[:batch_size], y[:batch_size]
