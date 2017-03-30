import tensorflow as tf

from DeepRL.learn import collections as cl


class Optimizer(object):

    def __init__(self, learning_rate, use_locking, name, step_tensor):
        self.step_tensor = step_tensor
        self.learning_rate = learning_rate
        self.use_locking = use_locking
        self.name = name
        self.tensor = None
        self.has_decay = False
        self.built = False

    def build(self):
        raise NotImplementedError

    def get_tensor(self):
        if not self.built:
            self.build()
        return self.tensor

    def minimize(self, loss):
        return self.get_tensor().minimize(loss, global_step=self.step_tensor)


class SGD(Optimizer):

    def __init__(self, learning_rate=0.001, lr_decay=0., decay_step=100,
                 staircase=False, use_locking=False, name="SGD", step_tensor=None):
        super(SGD, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step
        self.staircase = staircase

    def build(self):
        self.built = True
        if self.has_decay:
            if not self.step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.step_tensor,
                self.decay_step, self.lr_decay,
                staircase=self.staircase
            )
            tf.add_to_collection(cl.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            use_locking=self.use_locking,
            name=self.name
        )


class RMSProp(Optimizer):

    def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0,
                 epsilon=1e-10, use_locking=False, name="RMSProp", step_tensor=None):
        super(RMSProp, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self):
        self.built = True
        self.tensor = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate, decay=self.decay,
            momentum=self.momentum, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name
        )


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, use_locking=False, name="Adam", step_tensor=None):
        super(Adam, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build(self):
        self.built = True
        self.tensor = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1,
            beta2=self.beta2, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name
        )


class Momentum(Optimizer):

    def __init__(self, learning_rate=0.001, momentum=0.9, lr_decay=0.,
                 decay_step=100, staircase=False, use_locking=False,
                 name="Momentum", step_tensor=None):
        super(Momentum, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.momentum = momentum
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step
        self.staircase = staircase

    def build(self):
        self.built = True
        if self.has_decay:
            if not self.step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.step_tensor,
                self.decay_step, self.lr_decay,
                staircase=self.staircase)
            tf.add_to_collection(cl.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_locking=self.use_locking,
            name=self.name
        )


class AdaGrad(Optimizer):

    def __init__(self, learning_rate=0.001, initial_accumulator_value=0.1,
                 use_locking=False, name="AdaGrad", step_tensor=None):
        super(AdaGrad, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.initial_accumulator_value = initial_accumulator_value

    def build(self):
        self.built = True
        self.tensor = tf.train.AdagradOptimizer(
            self.learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            use_locking=self.use_locking, name=self.name
        )


class Ftrl(Optimizer):

    def __init__(self, learning_rate=3.0, learning_rate_power=-0.5,
                 initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0, use_locking=False,
                 name="Ftrl", step_tensor=None):
        super(Ftrl, self).__init__(learning_rate, use_locking, name, step_tensor)
        self.learning_rate_power = learning_rate_power
        self.initial_accumulator_value = initial_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength

    def build(self):
        self.built = True
        with tf.device('/cpu:0'):
            self.tensor = tf.train.FtrlOptimizer(
                self.learning_rate,
                learning_rate_power=self.learning_rate_power,
                initial_accumulator_value=self.initial_accumulator_value,
                l1_regularization_strength=self.l1_regularization_strength,
                l2_regularization_strength=self.l2_regularization_strength,
                use_locking=self.use_locking, name=self.name
            )