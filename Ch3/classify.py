import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

(tran_x, tran_y), (test_x, test_y) = datasets.mnist.load_data()

tran_x = 2*tf.convert_to_tensor(tran_x, dtype=tf.float32)/255. - 1
tran_y = tf.convert_to_tensor(tran_y, dtype=tf.int32)
tran_y = tf.one_hot(tran_y, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((tran_x, tran_y)) # 构建数据集对象
train_dataset = train_dataset.batch(512)