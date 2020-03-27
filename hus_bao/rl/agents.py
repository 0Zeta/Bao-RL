from random import choice, randint

import numpy as np
from scipy.special import softmax

from hus_bao.envs.hus_bao_env import HusBaoEnv
from hus_bao.rl.model import encode_states


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
            estimated_values = np.reshape(self.model.predict(encode_states(possible_states, 'test2')), newshape=(-1,))
            probabilities = softmax(estimated_values)
            return int(np.random.choice(available_actions, p=probabilities))


class AlphaBetaRLAgent(Agent):
    """a rl agent that uses an alpha beta-pruned search"""

    def __init__(self, model, exploration_rate):
        """
        Arguments:
            model (Model):            the model to use for predictions
            exploration_rate (float): the probability to choose a random move
        """
        self.model = model
        self.exploration_rate = exploration_rate
        self.env = HusBaoEnv()

    def move(self, game_state, available_actions):
        if randint(0, 100) <= self.exploration_rate * 100:
            return choice(available_actions)
        estimated_values = np.asarray([self._get_state_value(
            self.env.flip_board(self.env.get_board_after_action(action, game_state)), 0, False, 99999999, -99999999) for
                                       action in available_actions])
        probabilities = softmax(estimated_values)
        return int(np.random.choice(available_actions, p=probabilities))

    def _get_estimated_action_values(self, state):
        """estimates the values of all actions possible in the specified state
        Arguments:
            state (ndarray): the state that should be analyzed
        Returns:
            ndarray: the estimated values of all actions possible in the specified state
        """
        possible_states = np.reshape(np.asarray(
            [self.env.get_board_after_action(action, state) for action in self.env.get_available_actions(state)],
            dtype=np.int), newshape=(-1, 32))
        estimated_values = np.reshape(self.model.predict(encode_states(possible_states, 'test2')), newshape=(-1,))
        return estimated_values

    def _get_state_value(self, state, depth, maximizes, alpha, beta, max_depth=5):
        """estimates the value of a state
        Arguments:
            depth (int):      the current search depth
            state (ndarray):  the state that should be looked at
            maximizes (bool): whether the current node belongs to the maximizing player
            alpha (float):    alpha
            beta (float):     beta
            max_depth (int):  the maximum search depth
        """
        if state[2:].max() <= 1 or state[2].max() == 0:
            return 10 if maximizes else -10
        if state[0:1].max() <= 1 or state[1].max() == 0:
            return -10 if maximizes else 10
        if depth == max_depth:
            estimated_state_value = np.max(self._get_estimated_action_values(state))
            return estimated_state_value if maximizes else -estimated_state_value
        if maximizes:
            best_val = -10
            for child_state in [self.env.flip_board(self.env.get_board_after_action(action, state)) for action in
                                self.env.get_available_actions(state)]:
                value = self._get_state_value(child_state, depth + 1, False, alpha, beta)
                best_val = max(best_val, value)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val
        else:
            best_val = 10
            for child_state in [self.env.flip_board(self.env.get_board_after_action(action, state)) for action in
                                self.env.get_available_actions(state)]:
                value = self._get_state_value(child_state, depth + 1, True, alpha, beta)
                best_val = min(best_val, value)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val
