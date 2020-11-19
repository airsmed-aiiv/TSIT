from tsit.model.misc import FAdaIN, FADEResBlk
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GaussianNoise


def make_some_noise(shape, stddev=0.1, seed=None):
    if seed is None:
        return GaussianNoise(stddev=stddev)(tf.random.uniform(shape=shape))
    else:
        return GaussianNoise(stddev=stddev)(tf.random.uniform(shape=shape, seed=seed))

class Generator(Model):
    def __init__(self, k):
        self.faderesblk = []
        for i in range(7):
            self.faderesblk.append(FADEResBlk(k[i], k[i+1]))
    def call(self, cs, ss):
        j = 0
        z = make_some_noise(shape=[cs[-1].shape])
        for i in range(len(ss)-1, -1, -1):
            z = FAdaIN(z, ss[i])
            z = self.faderesblk[j](z, cs[i])
            j += 1
        return z

if __name__ == "__main__":
    print(make_some_noise(shape=[2, 2]))
