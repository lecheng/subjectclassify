import tensorflow as tf
from model import TextClassificationCNN
from config import TCCNNConfig
from data import *

def run_cnnclassify(is_train=True):
    model = TextClassificationCNN(TCCNNConfig,Subject())
    sess = tf.Session()
    if is_train:
        model.train(sess)
    else:
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(TCCNNConfig.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
        model.evaluate(sess)
    sess.close()

if __name__ == '__main__':
    run_cnnclassify(is_train=True)
