import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from vx_utils import show_loss_acc

DATA_GEN = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 20
TRIAN_SIZE = 100

CONV_BASE = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# 此算法数据较快，但不能使用图片增强算法


def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = DATA_GEN.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = CONV_BASE.predict(inputs_batch)
        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels


base_dir = "dogs_and_cats_small"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


train_features, train_labels = extract_features(train_dir, 20 * TRIAN_SIZE)
train_features = np.reshape(train_features, (20 * TRIAN_SIZE, 4 * 4 * 512))

validation_features, validation_labels = extract_features(validation_dir, 10 * TRIAN_SIZE)
validation_features = np.reshape(validation_features, (10 * TRIAN_SIZE, 4 * 4 * 512))

test_features, test_labels = extract_features(test_dir, 10 * TRIAN_SIZE)
test_features = np.reshape(test_features, (10 * TRIAN_SIZE, 4 * 4 * 512))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(
    train_features, train_labels,
    epochs=30,
    batch_size=BATCH_SIZE,
    validation_data=(validation_features, validation_labels)
)

show_loss_acc(history)
