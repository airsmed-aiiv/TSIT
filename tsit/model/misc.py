import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D
from tensorflow_addons.layers import InstanceNormalization


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


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
        self.ds1 = downsample(in_c, 4)
        self.csres1 = CSRes(in_c, 3)
        self.csres2 = CSRes(out_c, 3)
        self.csres3 = CSRes(out_c, 1)
    def call(self, x):
        x = self.ds1(x)
        x1, sc = self.csres1(x), self.csres2(x)
        x1 = self.csres3(x)
        x1 = tf.image.resize(x1, (sc.shape[1], sc.shape[2]))
        return tf.math.add(x1, sc)
    
def FAdaIN(content_features, style_features, alpha=1, epsilon = 1e-5):
    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keepdims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keepdims=True)
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features

class FADE(Model):
    def __init__(self, in_c):
        super(FADE, self).__init__()
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(in_c, 1, padding='same')
        self.conv2 = Conv2D(in_c, 1, padding='same')

    def call(self, x, feature):
        x = self.bn1(x)
        f1 = self.conv1(feature)
        f2 = self.conv2(feature)
        f1 = tf.image.resize(f1, (x.shape[1], x.shape[2]))
        f2 = tf.image.resize(f2, (x.shape[1], x.shape[2]))
        x *= f1
        x += f2
        return x


class FADERes(Model):
    def __init__(self, out_c, kernel):
        super(FADERes, self).__init__()
        self.lrelu = LeakyReLU(alpha=0.2)
        self.conv1 = Conv2D(out_c, kernel)
    def call(self, x):
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class FADEResBlk(Model):
    def __init__(self, in_c, out_c):
        super(FADEResBlk, self).__init__()
        self.fade1 = FADE(in_c)
        self.faderes1 = FADERes(in_c, 3)
        self.fade2 = FADE(in_c)
        self.faderes2 = FADERes(out_c, 3)
        self.fade3 = FADE(out_c)
        self.faderes3 = FADERes(out_c, 1)
        self.up1 = UpSampling2D(interpolation='bilinear')
    def call(self, x, feature):
        x1 = self.fade1(x, feature)
        x1 = self.faderes1(x1)
        sc = self.fade3(x, feature)
        sc = self.faderes3(x)
        x = self.fade2(x1, feature)
        x = self.faderes2(x)
        x = tf.image.resize(x, (sc.shape[1], sc.shape[2]))
        x = tf.math.add(x, sc)
        x = self.up1(x)
        return x
