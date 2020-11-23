import glob
import tensorflow as tf


def data2tfrecords(_a, _b, save_path):
    """Make common dataset as TFRecord.
    For Image-to-Image Translation task.

    Args:
        _a (list): List of A images location.
        _b (list): List of B images location.
        save_path (String): Where to save TFRecord file.
    """
    writer = tf.io.TFRecordWriter(save_path)
    for a, b in zip(_a, _b):
        a = open(a, 'rb').read()
        b = open(b, 'rb').read()
        sample = tf.train.Example(
            feature = {
                'a': tf.train.Feature(bytes_list=tf.train.BytesList(value=[a])),
                'b': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b])),
            }
        )
        writer.write(sample.SerializeToString())
        
if __name__ == "__main__":
    dataset_name = 'example'
    a_list = glob.glob('data/'+ dataset_name +'/a/*.png')
    b_list = glob.glob('data/'+ dataset_name +'/b/*.png')
    data2tfrecords(a_list, b_list, 'data/'+ dataset_name +'/')
