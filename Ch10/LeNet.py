import tensorflow as tf
import tensorflow.keras.datasets as datasets
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers

MODEL_SAVE_PATH = "letnet"

def preproces(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, (28,28, 1))
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


def load_data():
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).map(preproces).batch(200)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).map(preproces).batch(200)
    return train_dataset, test_dataset

def gen_network():
    network = Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(84, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(10)
        ])
    network.compile(
        optimizer = optimizers.Adam(lr=0.01),
        loss = tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return network

if __name__ == "__main__":
    train_dataset, test_dataset = load_data()
    network = gen_network()
    network.fit(train_dataset, epochs=3, validation_data=test_dataset, validation_freq=2)
    network.save(MODEL_SAVE_PATH)
    #network = keras.models.load_model(MODEL_SAVE_PATH)
    network.evaluate(test_dataset)