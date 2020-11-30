from tsit.model.misc import FAdaIN, FADEResBlk
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GaussianNoise


def make_some_noise(shape, stddev=0.1, seed=None):
    if seed is None:
        return GaussianNoise(stddev=stddev)(tf.random.uniform(shape=(shape[0][0], shape[0][1], shape[0][2], shape[0][3])))
    else:
        return GaussianNoise(stddev=stddev)(tf.random.uniform(shape=(shape[0], shape[1], shape[2], shape[3]), seed=seed))

class Generator(Model):
    def __init__(self, k):
        super(Generator, self).__init__()
        self.faderesblk = []
        for i in range(0, 7):
            self.faderesblk.append(FADEResBlk(k[i], k[i+1]))
    def call(self, cs, ss):
        z = make_some_noise(shape=[cs[-2].shape])
        for i in range(len(ss)-1, -1, -1):
            z = FAdaIN(z, ss[i])
            z = self.faderesblk[i](z, cs[i])
        return z

if __name__ == "__main__":
    print(make_some_noise(shape=[2, 2, 3, 4]))
