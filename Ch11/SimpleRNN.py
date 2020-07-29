import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers, losses, optimizers, Sequential

tf.random.set_seed(22)
np.random.seed(22)

BATCH_SIZE = 512
TOTAL_WORD = 10000
MAX_REVIEW_LEN = 80
EMBEDDING_LEN = 100

REVERSE_WORD_INDEX = None
WORD_INDEX = None
def load_word_index():
    global REVERSE_WORD_INDEX
    global WORD_INDEX

    if WORD_INDEX == None:
        word_index = keras.datasets.imdb.get_word_index()
        word_index = {k:(v+3) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3

    if REVERSE_WORD_INDEX == None:
        REVERSE_WORD_INDEX = {value: key for key, value in word_index.items()}

def decode_review(text):
    return " ".join([REVERSE_WORD_INDEX.get(i, '?') for i in text])

def load_data():
    (train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=TOTAL_WORD)
    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=MAX_REVIEW_LEN)
    test_x = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=MAX_REVIEW_LEN)

    train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_db = train_db.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
    test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_db = test_db.batch(BATCH_SIZE, drop_remainder=True)
    return train_db, test_db

class MyRNN(keras.Model):
    def __init__(self, units):
        super().__init__()
        self.embedding = layers.Embedding(TOTAL_WORD, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN)

        self.rnn = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(units, dropout=0.5)
            ])
        self.outlayer = Sequential([
            layers.Dense(32),
            layers.Dropout(rate=0.5),
            layers.ReLU(),
            layers.Dense(1)
            ])

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x, training)
        prob = tf.sigmoid(x)
        return prob

def main():
    units = 64
    epochs = 50

    train_db, test_db = load_data()

    model = MyRNN(units)
    model.compile(optimizer = optimizers.Adam(0.01),
        loss = losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    model.fit(train_db, epochs=epochs, validation_data=test_db)
    model.save("simple_rnn")

if __name__ == "__main__":
    main()