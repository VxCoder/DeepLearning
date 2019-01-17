from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def mnist_train():
    # load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # preparing the image data
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    # preparing the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # network architecture
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    # network compilation
    network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # network train
    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    # network test
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(f'test_acc: {test_acc}')


def show_digit():
    (train_images, train_labels), *_ = mnist.load_data()

    for show_num in range(0, 10):
        print(f"label{show_num}:{train_labels[show_num]}")
        plt.imshow(train_images[show_num], cmap=plt.cm.binary)
        plt.show()


def main():
    show_digit()


if __name__ == "__main__":
    main()
