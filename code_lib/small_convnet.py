from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

MODEL_SAVE_PATH = "number_model.h5"


def pre_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(f"train shape:{train_images.shape}")
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def trian_network():
    train_images, train_labels, test_images, test_labels = pre_data()
    model = build_model()
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    result = model.evaluate(test_images, test_labels)
    print(result)

    model.save(MODEL_SAVE_PATH)

    return model


if __name__ == "__main__":
    model = trian_network()
