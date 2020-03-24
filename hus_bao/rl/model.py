from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, \
    Flatten
from tensorflow.keras.models import Model


def build_model():
    """creates the model
    Returns:
        Model: the model
    """
    inputs = Input(shape=(32,))

    # fully connected network
    dense_1 = Dense(64, activation='relu')(inputs)
    dropout_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(96, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.2)(dense_2)
    dense_3 = Dense(128, activation='relu')(dropout_2)

    # Convnet
    reshape = Reshape(target_shape=(4, 8, 1))(inputs)
    conv_1 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid')(reshape)
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
