import random

import numpy as np


def random_shuffle(*args):
    index = list(range(len(args[0])))
    random.shuffle(index)
    ret = []
    for arg in args:
        if type(arg) == list:
            ret.append([arg[i] for i in index])
        else:
            ret.append(arg[index])
    return ret


def make_batches(samples_size, batch_size):
    nb_batch = int(np.ceil(samples_size/float(batch_size)))
    return [(i*batch_size, min(samples_size, (i+1)*batch_size)) for i in range(0, nb_batch)]