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

model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
    ])
optimizer = optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
    for step, (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())

def train():
    for epoch in range(30):
        train_epoch(epoch)

if __name__ == "__main__":
    train()