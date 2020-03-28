from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, \
    Flatten
from tensorflow.keras.models import Model


def build_model(architecture):
    """creates the model
    Arguments:
        architecture (str): the model architecture that should be used
    Returns:
        Model: the model
    """
    if architecture == 'test1':
        return _build_test1()
    elif architecture == 'test2':
        return _build_test2()
    elif architecture == 'test3':
        return _build_test3()


def encode_states(states, architecture):
    """Encodes the game state(s) to fit the specified architecture
    Arguments:
        states (ndarray):   the game state(s) that should get encoded
        architecture (str): the model architecture
    Returns:
        ndarray: the encoded game state(s)
    """
    if architecture == 'test1':
        return np.asarray(states, dtype=int)
    elif architecture == 'test2':
        return _encode_test2(states)
    elif architecture == 'test3':
        return _encode_test3(states)


def _build_test1():
    inputs = Input(shape=(32,))

    # fully connected network
    dense_1 = Dense(64, activation='relu')(inputs)
    dropout_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(96, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.2)(dense_2)
    dense_3 = Dense(128, activation='relu')(dropout_2)

    # Convnet
    reshape = Reshape(target_shape=(4, 8,))(inputs)
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(reshape)
    max_pooling_1 = MaxPooling2D(pool_size=2, strides=1, padding='valid')(conv_1)
    average_pooling_1 = AveragePooling2D(pool_size=2, strides=1, padding='valid')(conv_1)
    concatenate_1 = Concatenate(axis=1)([max_pooling_1, average_pooling_1])
    conv_2 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(concatenate_1)
    max_pooling_2 = MaxPooling2D(pool_size=3, strides=1, padding='valid')(conv_2)
    average_pooling_2 = AveragePooling2D(pool_size=3, strides=1, padding='valid')(conv_2)
    concatenate_2 = Concatenate(axis=1)([max_pooling_2, average_pooling_2])
    flatten = Flatten()(concatenate_2)
    conv_dense = Dense(256, activation='relu')(flatten)

    # Concatenate
    concatenate_3 = Concatenate(axis=1)([dense_3, conv_dense])
    dense_4 = Dense(400, activation='relu')(concatenate_3)
    dropout_3 = Dropout(0.1)(dense_4)
    dense_5 = Dense(256, activation='relu')(dropout_3)
    prediction = Dense(1)(dense_5)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def _build_test2():
    inputs = Input(shape=(768,))  # uses one-hot encoding capping all values above 24

    # fully connected network
    dense_1 = Dense(1024, activation='relu')(inputs)
    dense_2 = Dense(1024, activation='relu')(dense_1)
    dense_3 = Dense(700, activation='relu')(dense_2)
    dense_4 = Dense(512, activation='relu')(dense_3)
    dropout = Dropout(0.2)(dense_4)
    dense_5 = Dense(256, activation='relu')(dropout)
    prediction = Dense(1)(dense_5)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def _build_test3():
    inputs = Input(shape=(704,))  # uses one-hot encoding capping all values above 22

    dense_1 = Dense(800, activation='relu')(inputs)

    # Convnet
    reshape = Reshape(target_shape=(4, 8, 22))(inputs)
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(reshape)
    conv_2 = Conv2D(16, kernel_size=(6, 6), strides=1, padding='same', activation='relu')(conv_1)
    flatten = Flatten()(conv_2)

    concatenate = Concatenate()([dense_1, flatten])
    dense_2 = Dense(2048, activation='relu')(concatenate)
    dense_3 = Dense(1024, activation='relu')(dense_2)
    dropout = Dropout(0.2)(dense_3)
    dense_4 = Dense(512, activation='relu')(dropout)
    predictions = Dense(1)(dense_4)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def _encode_test2(states):
    encoded_states = np.ndarray(shape=(len(states), 768))
    for i, state in enumerate(states):  # TODO: optimize
        encoded_states[i] = _one_hot(state, 24)
    return encoded_states


def _encode_test3(states):
    encoded_states = np.ndarray(shape=(len(states), 704))
    for i, state in enumerate(states):  # TODO: optimize
        encoded_states[i] = _one_hot(state, 22)
    return encoded_states


def _one_hot(state, cap):
    """one-hot encoded a game state
    Arguments:
        state (ndarray): the game state that should be encoded
        cap (int):       the cap for stone counts
    Returns:
        ndarray: the encoded game states
    """
    encoded_state = np.zeros(shape=(32 * cap,), dtype=int)
    for i, field in enumerate(state):
        encoded_state[(i * cap + min(field, cap - 1))] = 1
    return encoded_state
