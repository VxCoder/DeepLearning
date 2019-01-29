import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb
from keras import regularizers
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):

    result = np.zeros((len(sequences), dimension))
    for index, sequence in enumerate(sequences):
        result[index, sequence] = 1

    return result


def show_loss_acc(history):
    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.plot(epochs, acc_values, 'y', label='Training acc')
    plt.plot(epochs, val_acc_values, 'g', label='Validation acc')
    plt.title('Training and validation loss/acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Acc')
    plt.legend()
    plt.show()


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(10000,)))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# history = network.fit(
#     partial_x_train,
#     partial_y_train,
#     epochs=20,
#     batch_size=512,
#     validation_data=(x_val, y_val)
# )
#
# show_loss_acc(history)


network.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=512,
)

results = network.evaluate(x_test, y_test)
print(results)
