import gym
import numpy as np
from gym import spaces

N_ROWS = 4
N_FIELDS = 8
N_ACTIONS = 16


class BaoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BaoEnv, self).__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=32, shape=(N_ROWS, N_FIELDS), dtype=int)
        self.state = np.asarray(N_ROWS * [N_FIELDS * [2]])
        self.done = False
        self.current_player = 0
        self.outcome = 0  # 0 if the game hasn´t ended yet, -1 if player 0 won, 1 if player 1 won

    def step(self, action):
        """steps the environment by an action.
        Args:
            action (int): the field to choose
        Returns:
            list:  observation
            float: reward
            bool:  done
            dict:  information
        """
        assert not self.done
        assert isinstance(action, int)
        assert action in self.get_available_actions()

        self._execute_action(action)

        self.done, self.outcome = self._check_winning_condition()
        data = np.copy(self.state), self._get_reward(), self.done, self._get_info()
        self.state = self.flip_board()
        self.current_player = (self.current_player + 1) % 2
        return data

    def reset(self):
        """resets the environment
        Returns:
            ndarray: observation
        """
        self.done = False
        self.state = np.asarray(N_ROWS * [N_FIELDS * [2]])
        self.current_player = 0
        return np.copy(self.state)

    def render(self, mode='human', close=False):
        """prints the game state to the console"""
        state = np.copy(self.state)
        print("")
        for row in reversed(range(N_ROWS)):
            print(str(np.reshape(np.flip(state[row], axis=0), (-1)).tolist()).replace('[', '').replace(']', '').replace(
                ',', ''))

    def close(self):
        return None

    def get_available_actions(self, state=None):
        """returns all actions the specified player can choose with the specified board
        Arguments:
            state (ndarray):  the board state
        Returns:
            list: actions the current player can choose
        """
        if state is None:
            state = np.copy(self.state)
        return np.reshape(np.argwhere(state[0] > 1), (-1)).tolist() + np.reshape(np.argwhere(state[1] > 1) + 8,
                                                                                 (-1)).tolist()

    def get_board_after_action(self, action, state=None):
        """returns the board state after executing the specified action in the specified state
        Arguments:
            action (int):    the field to choose
            state (ndarray): the board state
        Returns:
            ndarray: the board state after the action
        """
        if state is None:
            state = self.state
        return self._execute_action(action, np.copy(state))

    def _execute_action(self, action, state=None):
        """executes the action for the specified state (changes the state argument)
        Arguments:
            action (int):    the field to choose
            state (ndarray): the board state
        Returns:
            ndarray: the board state after the action
        """
        if state is None:
            state = self.state
        row, field = self.get_coordinates(action)
        n_stones = state[row, field]
        state[row, field] = 0
        while n_stones > 0:
            if row == 0:
                if field + 1 < N_FIELDS:
                    field += 1
                else:
                    row = 1
                    # field stays the same
            elif row == 1:
                if field - 1 >= 0:
                    field -= 1
                else:
                    row = 0
                    # field stays the same
            state[row, field] += 1
            n_stones -= 1

        if row == 1 and state[2, field] > 0:
            state[row, field] += state[2, field]
            state[2, field] = 0
        if state[row, field] > 1:
            done, outcome = self._check_winning_condition(state)
            if done:
                return state
            self._execute_action(field + (8 if row == 1 else 0), state)
        return state

    def _get_reward(self):
        """returns the reward for the current player
        Returns:
            int: reward
        """
        if self.outcome == 0:
            return 0
        else:
            if (self.outcome == -1 and self.current_player == 0) or (
                    self.outcome == 1 and self.current_player == 1):
                return 1
            return -1

    def _get_info(self):
        """returns some additional information
        Returns:
            dict: info on the game
        """
        return {'current_player': self.current_player,
                'outcome': self.outcome}

    def _check_winning_condition(self, state=None, current_player=None):
        """checks whether the game has ended yet
        Arguments:
            state (ndarray):      the current board state
            current_player (int): the current player
        Returns:
            bool: done
            int: outcome
        """
        if state is None:
            state = self.state
        if current_player is None:
            current_player = self.current_player
        if state[2:].max() <= 1 or state[2].max() == 0:
            return True, -1 if current_player == 0 else 1
        return False, 0

    def flip_board(self, state=None):
        """flips the board
        Arguments:
            state (ndarray): the board state
        Returns:
            ndarray: the flipped board
        """
        if state is None:
            state = self.state
        return np.flip(np.flip(state, axis=0), axis=1)

    @staticmethod
    def get_coordinates(action):
        """returns row and field for a given action
        Arguments:
            action (int): the action
        Returns:
            (int, int): row, field
        """
        row = 0 if action < N_FIELDS else 1
        field = action % N_FIELDS
        return row, field