import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


ORIGINAL_DATASET_DIR = "dogs-vs-cats/train"
BASE_DIR = "dogs_and_cats_small"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')


def files_copy(fnames, dst_dir):
    for fname in fnames:
        src = os.path.join(ORIGINAL_DATASET_DIR, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)


def pre_train_data():
    os.mkdir(BASE_DIR)

    # 训练集
    train_dir = TRAIN_DIR
    os.mkdir(train_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    fnames = [f'cat.{i}.jpg' for i in range(1000)]
    files_copy(fnames, train_cats_dir)
    fnames = [f'dog.{i}.jpg' for i in range(1000)]
    files_copy(fnames, train_dogs_dir)

    # 验证集
    validation_dir = VALIDATION_DIR
    os.mkdir(validation_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
    files_copy(fnames, validation_cats_dir)
    fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
    files_copy(fnames, validation_dogs_dir)

    # 测试集
    test_dir = os.path.join(BASE_DIR, 'test')
    os.mkdir(test_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)
    fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
    files_copy(fnames, test_cats_dir)
    fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
    files_copy(fnames, test_dogs_dir)


def build_network():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc']
    )

    return model


def pre_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    validation_datagen = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_datagen


def train_network(model, train_generator, validation_generator):

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save('cats_and_dogs_small_2.h5')

    return history


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


def change_image():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_cats_dir = os.path.join(TRAIN_DIR, 'cats')
    fnames = [os.path.join(train_cats_dir, fname) for
              fname in os.listdir(train_cats_dir)]

    img_path = fnames[5]
    img = image.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)

    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 5 == 0:
            break
    plt.show()


if __name__ == "__main__":
    # change_image()
    # pre_train_data()
    model = build_network()
    train_generator, validation_datagen = pre_data()
    history = train_network(model, train_generator, validation_datagen)
    show_loss_acc(history)
