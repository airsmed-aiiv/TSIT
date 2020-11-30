import tensorflow as tf
from tsit.model.TSIT import TSIT


if __name__ == "__main__":
    model = TSIT([64, 128, 256, 512, 1024, 1024, 1024, 1024])
    output = model(tf.zeros([1, 1024, 1024, 3]))
    model.save('model')
    