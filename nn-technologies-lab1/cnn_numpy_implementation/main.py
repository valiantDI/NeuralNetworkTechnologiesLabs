import numpy as np
import json
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.metrics import classification_report
from linear import Linear
from convolutional import Convolutional
from activations import Sigmoid, Softmax
from reshape import Reshape
from losses import cross_entropy, cross_entropy_derivative
from train import train, predict


def preprocess_data(x, y, limit, train_size=0.8, sample='train'):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    tree_index = np.where(y == 3)[0][:limit]
    four_index = np.where(y == 4)[0][:limit]
    five_index = np.where(y == 5)[0][:limit]
    six_index = np.where(y == 6)[0][:limit]
    seven_index = np.where(y == 7)[0][:limit]
    eight_index = np.where(y == 8)[0][:limit]
    nine_index = np.where(y == 9)[0][:limit]
    all_indices = np.hstack((zero_index, one_index, two_index, tree_index, four_index, five_index, six_index,
                             seven_index, eight_index, nine_index))
    all_indices = np.random.permutation(all_indices)

    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)


network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    # Convolutional((10, 26, 26), fold3, 10),
    # Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Linear(5 * 26 * 26, 100),
    Sigmoid(),
    Linear(100, 10),
    Softmax()
]

train_errors, test_errors = train(
                                 network,
                                 cross_entropy,
                                 cross_entropy_prime,
                                 x_train,
                                 y_train,
                                 x_test,
                                 y_test,
                                 epochs=50,
                                 learning_rate=0.001
                                 )

y_pred = []
y_true = []

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    y_pred.append(np.argmax(output))
    y_true.append(np.argmax(y))

print(classification_report(y_true, y_pred))

with open("train_errors.json", "w") as file:
    json.dump(train_errors, file, indent=2)

with open("test_errors.json", "w") as file:
    json.dump(val_errors, file, indent=2)
