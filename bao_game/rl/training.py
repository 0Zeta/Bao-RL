import random
from random import choice, getrandbits

import gym
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint

from bao_game.envs.bao_utils import *
from bao_game.rl.agents import Agent, MinimaxRLAgent
from bao_game.rl.model import build_model, encode_states

RANDOM_STATE = 12345
random.seed(RANDOM_STATE)
ARCHITECTURE = 'test2'


def data_generator(batch_size, agent: Agent, opponents, model, gamma=0.92, move_penalty=-0.002):
    """generates data for the model
    Arguments:
        batch_size (int):     how many datapoints should be generated
        agent (Agent):        the main agent for the data
        opponents (list):     the opponents for the agent
        model (Model):        the model that should be used to evaluate states
        gamma (float):        the discount factor for the added value estimate of the next state
        move_penalty (float): the reward that should be added to each move
    Returns:
        ndarray: the game states (after an action)
        ndarray: the computed values of the states
    """
    env = gym.make('Bao-v0')
    while True:
        X = []
        y = []

        while len(X) < batch_size:
            env.reset()
            opponent = choice(opponents)

            if getrandbits(1) == 1:
                current_agent = agent  # the main agent goes first
                waiting_agent = opponent
            else:
                current_agent = opponent
                waiting_agent = agent

            states_after_action = [[], []]
            while not env.done:
                state = np.copy(env.state)
                current_player = env.current_player
                next_state, reward, done, _ = env.step(
                    current_agent.move(state, get_available_actions(state=env.state)))
                states_after_action[current_player].append(next_state)
                current_agent, waiting_agent = waiting_agent, current_agent
            # Compute state values (The states are the results of actions.)
            values = [np.full(fill_value=move_penalty, shape=(len(states_after_action[0]),), dtype=np.float),
                      np.full(fill_value=move_penalty, shape=(len(states_after_action[1]),), dtype=np.float)]
            outcome = env.outcome
            values[0][-1] += -outcome * 10
            values[1][-1] += outcome * 10

            # Add estimates of the new states to every but the last states (=> [:-1])
            flipped_states = [[flip_board(state) for state in states_after_action[i][:-1]] for i in range(2)]
            possible_states = [[[np.reshape(get_board_after_action(action, state), newshape=(32,)) for action in
                                 get_available_actions(state)] for state in flipped_states[i]] for i in range(2)]
            estimates = [
                [np.reshape(model.predict(encode_states(x, ARCHITECTURE)), newshape=(-1,)) for x in possible_states[i]]
                for
                i in range(2)]  # TODO: optimize
            next_state_values = [np.asarray([-np.max(x) for x in estimates[i]], dtype=np.float) for i in range(2)]

            for z in range(2):
                values[z][:-1] += gamma * next_state_values[z]
                X.extend([np.reshape(state, newshape=(32,)) for state in states_after_action[z]])
                y.extend(values[z])
        X = encode_states(X, ARCHITECTURE)
        X, y = shuffle(X, np.asarray(y), random_state=RANDOM_STATE)
        yield X[:batch_size], y[:batch_size]


def train_model():
    model = build_model(ARCHITECTURE)
    model.load_weights("F:/model_checkpoints/bao_game/test2/weights.150-0.99.hdf5")
    model.fit_generator(
        generator=data_generator(200, MinimaxRLAgent(model, exploration_rate=0.01, min_prob=0.01), [
            MinimaxRLAgent(model, exploration_rate=0.05, min_prob=0.05)
        ], model),
        callbacks=[ModelCheckpoint(
            filepath='F:/model_checkpoints/bao_game/test2/weights.{epoch:02d}-{loss:.2f}.hdf5',
            monitor='loss', save_weights_only=True, save_best_only=False, save_freq=10 * 200)],
        epochs=1000000, steps_per_epoch=1, initial_epoch=150)


if __name__ == '__main__':
    train_model()
