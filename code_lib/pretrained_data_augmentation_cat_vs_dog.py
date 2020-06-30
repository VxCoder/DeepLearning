import os
from keras import models
from keras import layers


from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# 该算法允许使用图像增强算法,但需要大量训练时间

BASE_DIR = "dogs_and_cats_small"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')


def build_model():

    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    conv_base.trainable = False   # 冻结,防止权重数据丢失

    # 微调技术
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True

        layer.trainable = set_trainable

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=['acc']
    )

    return model


if __name__ == "__main__":
    model = build_model()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=2,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
