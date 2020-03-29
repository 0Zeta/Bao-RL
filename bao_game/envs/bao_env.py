import gym
from gym import spaces

from bao_game.envs.bao_utils import *

N_ROWS = 4
N_FIELDS = 8
N_ACTIONS = 16


class BaoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BaoEnv, self).__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=35, shape=(N_ROWS, N_FIELDS), dtype=int)
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
        assert action in get_available_actions(self.state)

        execute_action(state=self.state, action=action)

        self.done, self.outcome = check_winning_condition(state=self.state, current_player=self.current_player)
        data = np.copy(self.state), self._get_reward(), self.done, self._get_info()
        self.state = flip_board(self.state)
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
