import tensorflow as tf


def is_training(is_training=False,  session=None):
    if not session:
        session = tf.get_default_session()
    init_training_mode()
    if is_training:
        tf.get_collection('is_training_ops')[0].eval(session=session)
    else:
        tf.get_collection('is_training_ops')[1].eval(session=session)


def get_training_mode():
    init_training_mode()
    coll = tf.get_collection('is_training')
    return coll[0]


def init_training_mode():
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        tr_var = tf.get_variable(
            "is_training", dtype=tf.bool, shape=[],
            initializer=tf.constant_initializer(False),
            trainable=False
        )
        tf.add_to_collection('is_training', tr_var)
        # 'is_training_ops' stores the ops to update training mode variable
        a = tf.assign(tr_var, True)
        b = tf.assign(tr_var, False)
        tf.add_to_collection('is_training_ops', a)
        tf.add_to_collection('is_training_ops', b)