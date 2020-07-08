import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from    tensorflow import keras

BATCH_SIZE = 128
def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

def load_data():
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    train_db = train_db.map(preprocess).shuffle(60000).batch(BATCH_SIZE)
    test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_db = test_db.map(preprocess).batch(BATCH_SIZE) 
    return train_db, test_db

class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim], trainable=True)

    def call(self, inputs, training=None):
        out = inputs @ self.kernel
        out = tf.nn.relu(out)
        return out

class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        # 自定义前向运算逻辑
        x = self.fc1(inputs) 
        x = self.fc2(x) 
        x = self.fc3(x) 
        x = self.fc4(x) 
        x = self.fc5(x) 
        return x


def main():
    train_db, test_db = load_data()
    network = MyModel()
    network.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    network.fit(train_db, epochs=5, validation_data=test_db, validation_freq=2)
    network.save("self_define_model")
    network.evaluate(test_db)


if __name__ == "__main__":
    main()