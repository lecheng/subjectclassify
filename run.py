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
        model.evaluate(sess)
    sess.close()

if __name__ == '__main__':
    run_cnnclassify(is_train=True)