import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)

x = tf.random.normal([4,224,224, 3])
out = resnet(x)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
fc = layers.Dense(100)

mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()
resnet.trainable = False
mynet.summary()