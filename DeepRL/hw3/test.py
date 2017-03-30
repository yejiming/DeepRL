import tensorflow as tf

from DeepRL.hw3 import run_dqn_ram
from DeepRL.hw3.dqn_utils import *


env = run_dqn_ram.get_env(0)
frame_history_len = 4
gamma = 0.99

if len(env.observation_space.shape) == 1:
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_shape = env.observation_space.shape
else:
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
num_actions = env.action_space.n

act_t_ph = tf.placeholder(tf.int32, [None])
obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
rew_t_ph              = tf.placeholder(tf.float32, [None])
obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
done_mask_ph          = tf.placeholder(tf.float32, [None])

obs_t_float = tf.cast(obs_t_ph,   tf.float32) / 255.0
obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

q = run_dqn_ram.atari_model(obs_t_float, num_actions, scope="q_func", reuse=False)
target_q = run_dqn_ram.atari_model(obs_t_float, num_actions, scope="target_q_func", reuse=False)
q_tp1 = tf.reduce_max(run_dqn_ram.atari_model(obs_tp1_float, num_actions, scope="target_q_func", reuse=False))

session = tf.Session()
session.run(tf.global_variables_initializer())

current_Q = tf.reduce_sum(tf.gather(target_q, act_t_ph), axis=1)
next_Q = rew_t_ph + gamma * q_tp1 * (1 - done_mask_ph)
total_error = tf.reduce_sum(huber_loss(tf.square(current_Q - next_Q) / 2))

x = np.asarray([[1,2,3,3,2,5,6,7,1,3], [1,2,3,3,2,5,6,7,1,3]])
e = np.asarray([[2], [2]])
x_t = tf.constant(x)
e_t = tf.constant(e)
e_t = tf.reshape(tf.one_hot(e_t, 10), [-1, 10])
e_t = tf.cast(e_t, tf.bool)

result = tf.boolean_mask(x_t, e_t)
print(session.run(result))