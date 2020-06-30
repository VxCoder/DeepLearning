import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from vx_utils import show_loss_acc


def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for index, sequence in enumerate(sequences):
        result[index, sequence] = 1.

    return result


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))

    for index, label in enumerate(labels):
        results[index, label] = 1.
    return results


def train_analysis(x_train, y_labels, network):
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_labels[:1000]
    partial_y_train = y_labels[1000:]

    history = network.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val))

    show_loss_acc(history)


(train_data, train_labels), (test_data, test_labes) = reuters.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_trian_labels = to_one_hot(train_labels)  # keras.utils.np_utils.to_categorical
one_hot_test_labels = to_one_hot(test_labes)  # keras.utils.np_utils.to_categorical

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(46, activation='softmax'))

network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_analysis(x_train, one_hot_trian_labels, network)
# network.fit(
#     x_train,
#     one_hot_trian_labels,
#     epochs=5,
#     batch_size=512
# )
# results = network.evaluate(x_test, one_hot_test_labels)

# print(results)
