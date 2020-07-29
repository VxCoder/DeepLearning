import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers,Sequential

BATCH_SIZE = 512
H_DIM = 20
LR = 1e-3

def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

def preprocess(x):
    x = tf.cast(x, np.float32) / 255.
    return x

def load_data():
    (x_trian, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices(x_trian)
    train_db = train_db.map(preprocess).shuffle(BATCH_SIZE * 5).batch(BATCH_SIZE)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.map(preprocess).batch(BATCH_SIZE)
    return train_db, test_db


class AE(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(H_DIM)
        ])

        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
            ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)

        return x_hat

def main():
    train_db, test_db = load_data()

    model = AE()
    model.build(input_shape=(4, 784))
    optimizer = tf.optimizers.Adam(lr=LR)

    for epoch in range(100):
        for step, x in enumerate(train_db):
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                x_rec_logits = model(x)
                rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)

            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, float(rec_loss))

            x = next(iter(test_db))
            logtis = model(tf.reshape(x, [-1, 784]))
            x_hat = tf.sigmoid(logtis)
            x_hat = tf.reshape(x_hat, [-1, 28, 28])
            x_concat = tf.concat([x, x_hat], axis=0)
            x_concat = x_hat
            x_concat = x_concat.numpy() * 255.
            x_concat = x_concat.astype(np.uint8)
            save_images(x_concat, 'ae_images/rec_epoch_%d.png'%epoch)

if __name__ == "__main__":
    main()