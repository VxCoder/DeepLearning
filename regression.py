from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np


def normal_data(data, mean=None, std=None):
    mean = mean if mean is not None else data.mean(axis=0)
    data -= mean
    std = std if std is not None else data.std(axis=0)
    data /= std

    return data, mean, std


def get_data():
    (train_data, trian_targets), (test_data, test_targets) = boston_housing.load_data()
    train_data, mean, std = normal_data(train_data)
    test_data = normal_data(test_data, mean, std)

    return train_data, trian_targets, test_data, test_targets


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mae']
    )

    return model


def validation(train_data, trian_targets):
    k = 4

    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):
        print("processing fold #", i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = trian_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[: i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0
        )

        partial_train_targets = np.concatenate(
            [trian_targets[: i * num_val_samples],
             trian_targets[(i + 1) * num_val_samples:]],
            axis=0
        )

        model = build_model()

        model.fit(
            partial_train_data,
            partial_train_targets,
            epochs=num_epochs,
            batch_size=1,
            verbose=0
        )

        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

    print(all_scores)


train_data, trian_targets, test_data, test_targets = get_data()
validation(train_data, trian_targets)
