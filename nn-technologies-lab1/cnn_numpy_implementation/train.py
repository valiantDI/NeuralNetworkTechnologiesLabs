from collections import defaultdict


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, x_val=None, y_val=None, epochs=1000, learning_rate=0.01,
          verbose=True):
    train_errors = defaultdict(list)
    val_errors = defaultdict(list)
    for e in range(epochs):
        train_error = 0
        val_error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            train_error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        train_error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={train_error}")

        train_errors['epochs'].append(e + 1)
        train_errors['errors'].append(train_error)

        if x_val and y_val:
            for x, y in zip(x_val, y_val):
                # forward
                output = predict(network, x)

                # error
                val_error += loss(y, output)

            val_error /= len(x_val)
            if verbose:
                print(f"{e + 1}/{epochs}, error={val_error}")
            val_errors['epochs'].append(e + 1)
            val_errors['errors'].append(val_error)
    return train_errors, val_errors
