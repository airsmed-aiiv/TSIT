import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization

def bicubic_kernel(x, a=-0.5):
    if abs(x) <= 1:
        return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
    elif 1 < abs(x) and abs(x) < 2:
        return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a 
    else:
        return 0
    
def build_filter(factor):
    size = factor*4
    k = np.zeros((size))
    for i in range(size):
        x = (1/factor)*(i- np.floor(size/2) +0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    k = np.outer(k, k.T)
    k = tf.constant(k, dtype=tf.float32, shape=(size, size, 1, 1))
    return tf.concat([k, k, k], axis=2)

def apply_bicubic_downsample(x, filter, factor):
    filter_height = factor*4
    filter_width = factor*4
    strides = factor
    pad_along_height = max(filter_height - strides, 0)
    pad_along_width = max(filter_width - strides, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
    x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1,strides,strides,1], padding='VALID')
    return x

class CSRes(Model):
    def __init__(self, out_c, kernel):
        super(CSRes, self).__init__()
        self.conv1 = Conv2D(out_c, kernel)
        self.in1 = InstanceNormalization()
        self.lrelu = LeakyReLU(alpha=0.2)
    def call(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.lrelu(x)
        return x


class CSResBlk(Model):
    def __init__(self, in_c, out_c):
        super(CSResBlk, self).__init__()
        self.ds1 = build_filter(2)
        self.csres1 = CSRes(in_c, 3)
        self.csres2 = CSRes(out_c, 3)
        self.csres3 = CSRes(out_c, 1)
    def call(self, x):
        x = apply_bicubic_downsample(x, self.ds1, 4)
        x1, sc = self.csres1(x), self.csres2(x)
        x1 = self.csres3(x)
        return tf.math.add(x1, sc)
    

class FADE(Model):
    def __init__(self):
        super(FADE, self).__init__()

    def call(self, x, feature):
        pass


class FADEResBlk(Model):
    def __init__(self):
        super(FADEResBlk, self).__init__()

    def call(self, x, feature):
        pass
