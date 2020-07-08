import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets


def init_paramaters():
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3

def load_data():
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    train_x = 2* tf.convert_to_tensor(train_x, dtype=tf.float32) / 255. - 1
    train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
    train_y  = tf.one_hot(train_y, depth=10)

    train_x = tf.reshape(train_x, (-1, 28*28))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.batch(200)
    return train_dataset

def init_show():
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = ['STKaiti']
    plt.rcParams['axes.unicode_minus'] = False

def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.01):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x@w1 + b1)
            h2 = tf.nn.relu(h1@w2 + b2)
            out = h2@w3 + b3

            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())

    return loss.numpy()


def train(epochs):
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramaters()
    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.01)
        losses.append(loss)

    plt.plot(list(range(epochs)), losses, color='blue', marker='s', label='训练')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("MNIST数据集前向传播训练误差曲线")
    plt.close()

def main():
    init_show()
    train(epochs=20)
    print("work over")

if __name__ == "__main__":
    main()