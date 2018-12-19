import os, sys
from load_param import *
import tensorflow as tf

class FileExistsError(Exception):
    pass

def try_mkdir(dir, warning=True):
    try:
        if os.path.exists(dir):
            raise FileExistsError
        os.makedirs(dir)
    except FileExistsError:
        if(warning):
            print("Warning: dir " + dir + " already exist! Continue program...")
    except:
        print("Cannot make dir: " + dir)
        print(sys.exc_info()[0])
        exit(0)

def _parse_function(example_proto):
    feature = {'data': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    x = tf.decode_raw(parsed_features['data'],tf.float32)
    x = tf.reshape(x,[win_size,mfcc_dim])
    x = (x-mean)/std
    y = tf.cast(parsed_features['label'], tf.int32)
    return x, tf.one_hot(y,n_phoneme)

def add_summary(tag,value,writer,step):
    with tf.name_scope('test' + '_tensorboard'):
        summary = tf.Summary()
        summary.value.add(tag = tag, simple_value=value)
    writer.add_summary(summary, step)


if __name__ == "__main__":

    sess = tf.Session()
    with tf.name_scope("XiaoGongWei"):
        a = tf.placeholder(dtype=tf.float32)
        b = tf.Variable([1.0,2.0], dtype=tf.float32)
        W = tf.Variable([1, 1], dtype=tf.float32)
        addAB = W * a + b
    tf.summary.scalar("a_value",tf.reduce_mean(a))
    tf.summary.scalar("wight_max", tf.reduce_mean(W))
    tf.summary.scalar("b_value", tf.reduce_mean(b))
    merged = tf.summary.merge_all()

    train_summary = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        myadd = sess.run([addAB], feed_dict={a: i})
        #print(tfmensumm)
        summary = tf.Summary()
        summary.value.add(tag='test_loss',simple_value=i)
        train_summary.add_summary(summary, i)
    train_summary.close()





