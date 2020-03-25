from random import choice


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
