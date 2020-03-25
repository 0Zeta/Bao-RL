from random import choice, randint

import numpy as np
from scipy.special import softmax

from hus_bao.envs.hus_bao_env import HusBaoEnv


class Agent(object):
    def move(self, game_state, available_actions):
        """the agent has to decide on a move
        Arguments:
            game_state (ndarray):     the current game state
            available_actions (list): a list of the available moves
        Returns:
            int: a valid action
        """
        pass


class RandomAgent(Agent):
    """an agent that always chooses a random action"""

    def move(self, game_state, available_actions):
        return choice(available_actions)


class MostStonesAgent(Agent):
    """an agent that always chooses the action that uses the field with the most stones"""

    def move(self, game_state, available_actions):
        max_action = available_actions[0]
        max_stones = 0
        for action in available_actions:
            row, field = HusBaoEnv.get_coordinates(action)
            if game_state[row][field] > max_stones:
                max_action = action
                max_stones = game_state[row][field]
        return max_action


class SimpleRLAgent(Agent):
    """a simple rl agent that doesn´t look at future moves"""

    def __init__(self, model, exploration_rate, env):
        """
        Arguments:
            model (Model):            the model to use for predictions
            exploration_rate (float): the probability to choose a random move
            env (HusBaoEnv):          a game environment
        """
        self.model = model
        self.exploration_rate = exploration_rate
        self.env = env

    def move(self, game_state, available_actions):
        if randint(0, 100) <= self.exploration_rate * 100:
            return choice(available_actions)
        else:
            possible_states = np.reshape(
                np.asarray([self.env.get_board_after_action(action, game_state) for action in available_actions],
                           dtype=np.int), newshape=(-1, 32))
            estimated_values = np.reshape(self.model.predict(possible_states), newshape=(-1,))
            probabilities = softmax(estimated_values)
            return int(np.random.choice(available_actions, p=probabilities))
