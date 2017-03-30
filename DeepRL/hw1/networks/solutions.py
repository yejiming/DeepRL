import numpy as np
import tensorflow as tf
from DeepRL.learn import summaries
from sklearn.preprocessing import StandardScaler

from DeepRL.hw1.conf import constants
from DeepRL.hw1.networks import custom
from DeepRL.hw1.utils import data_utils
from DeepRL.learn import config as cfg


class SimpleSolution:

    def __init__(self, num_features, num_outputs):
        self.X = tf.placeholder(tf.float32, shape=[None, num_features])
        self.y = tf.placeholder(tf.float32, shape=[None, num_outputs])
        self.scaler = StandardScaler()

        self.pred_op, loss, self.train_op = custom.simple_model(self.X, self.y, num_outputs)
        tf.summary.scalar("loss", loss)
        self.summary = tf.summary.merge_all()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def fit(self, data):
        observations, actions = data["observations"], data["actions"]
        observations = self.scaler.fit_transform(observations)

        for i in range(constants.N_EPOCHS):
            observations, actions = data_utils.random_shuffle(observations, actions)

            samples_size = len(observations)
            batch_size = constants.BATCH_SIZE
            batches = data_utils.make_batches(samples_size, batch_size)

            for j, (start_index, end_index) in enumerate(batches):
                cfg.is_training(True, session=self.session)

                labels = np.array(actions[start_index:end_index])
                features = observations[start_index:end_index]

                feed_dict = {self.X: features, self.y: labels}
                _, summary_str = self.session.run([self.train_op, self.summary], feed_dict=feed_dict)
                loss = summaries.get_value_from_summary_string("loss", summary_str)

            print("Epoch: %d, Loss: %0.3f" % (i+1, loss))

    def predict(self, observation):
        observation = observation.reshape(1, -1)
        observation = self.scaler.transform(observation)
        feed_dict = {self.X: observation}
        pred = self.session.run(self.pred_op, feed_dict=feed_dict)
        return pred
