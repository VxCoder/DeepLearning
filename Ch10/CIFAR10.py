import tensorflow as tf
from tensorflow.keras import Sequential, layers, datasets, optimizers

def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def load_data():
    (train_x, train_y), (test_x, test_y) = datasets.cifar100.load_data()
    train_y = tf.squeeze(train_y, axis=1)
    test_y = tf.squeeze(test_y, axis=1)

    train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_db = test_db.map(preprocess).batch(128)

    return train_db, test_db

def get_fc_network():
    return Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None)
    ])

def gen_conv_network():
    return Sequential([
        layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=3, padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),
        ])

def main():
    train_db, test_db = load_data()

    conv_network = gen_conv_network()
    conv_network.build(input_shape=[None,32,32,3])
    fc_network = get_fc_network()
    fc_network.build(input_shape=[None,512])

    optimizer = optimizers.Adam(lr=1e-4)

    variables = conv_network.trainable_variables + fc_network.trainable_variables

    for epoch in range(50):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_network(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_network(out)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for x,y in test_db:

            out = conv_network(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_network(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

if __name__ == "__main__":
    main()

一致性hash实现
HBase