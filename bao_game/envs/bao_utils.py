import numpy as np


def get_available_actions(state):
    """returns all actions the specified player can choose with the specified board
    Arguments:
        state (ndarray):  the board state
    Returns:
        list: actions the current player can choose
    """
    return np.reshape(np.argwhere(state[0] > 1), (-1)).tolist() + np.reshape(np.argwhere(state[1] > 1) + 8,
                                                                             (-1)).tolist()


def get_board_after_action(action, state):
    """returns the board state after executing the specified action in the specified state
    Arguments:
        action (int):    the field to choose
        state (ndarray): the board state
    Returns:
        ndarray: the board state after the action
    """
    return execute_action(action, np.copy(state))


def execute_action(action, state):
    """executes the action for the specified state (changes the state argument)
    Arguments:
        action (int):    the field to choose
        state (ndarray): the board state
    Returns:
        ndarray: the board state after the action
    """
    row, field = get_coordinates(action)
    n_stones = state[row, field]
    state[row, field] = 0
    while n_stones > 0:
        if row == 0:
            if field + 1 < 8:
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
        done, outcome = check_winning_condition(state)
        if done:
            return state
        execute_action(field + (8 if row == 1 else 0), state)
    return state


def check_winning_condition(state, current_player):
    """checks whether the game has ended yet
    Arguments:
        state (ndarray):      the current board state
        current_player (int): the current player
    Returns:
        bool: done
        int: outcome
    """
    if state[2:].max() <= 1 or state[2].max() == 0:
        return True, -1 if current_player == 0 else 1
    return False, 0


def flip_board(state):
    """flips the board
    Arguments:
        state (ndarray): the board state
    Returns:
        ndarray: the flipped board
    """
    return np.flip(np.flip(state, axis=0), axis=1)


def get_coordinates(action):
    """returns row and field for a given action
    Arguments:
        action (int): the action
    Returns:
        (int, int): row, field
    """
    row = 0 if action < 8 else 1
    field = action % 8
    return row, field
