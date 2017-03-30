from DeepRL.learn import layers
from DeepRL.learn import objectives

from DeepRL.hw1.conf import constants
from DeepRL.learn import optimizers


def simple_model(features, labels, num_outputs):
    network = layers.fully_connected(features, 200, activation="relu", name="fc1")
    network = layers.dropout(network, 0.5)
    network = layers.fully_connected(network, 200, activation="relu", name="fc2")
    network = layers.dropout(network, 0.5)
    network = layers.fully_connected(network, 100, activation="relu", name="fc3")
    network = layers.dropout(network, 0.5)

    outputs = layers.fully_connected(network, n_units=num_outputs, activation="linear", name="output")
    loss = objectives.mean_square(outputs, labels)

    train_op = optimizers.Momentum(constants.LEARNING_RATE, 0.9).minimize(loss)

    return outputs, loss, train_op
