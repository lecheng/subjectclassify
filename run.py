import tensorflow as tf
from model import TextClassificationCNN
from embedding.word2vec_optimized import Word2Vec
from config import TCCNNConfig
from data import *

def run_cnnclassify(is_train=True):
    if is_train:
        g_1 = tf.Graph()
        with g_1.as_default():
            sess1 = tf.Session()
            checkpoint = tf.train.latest_checkpoint(TCCNNConfig.embedding_dir)
            embedding_saver = tf.train.import_meta_graph(checkpoint+'.meta')
            if checkpoint:
                embedding_saver.restore(sess1, checkpoint)
            embeddings = sess1.run('w_in:0')
            print embeddings.shape

        g_2 = tf.Graph()
        with g_2.as_default():
            model = TextClassificationCNN(TCCNNConfig, Subject())
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(model.embedding_init, feed_dict={model.embedding_placeholder:embeddings})
            model.train(sess)
    else:
        model = TextClassificationCNN(TCCNNConfig, Subject())
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(TCCNNConfig.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
        model.evaluate(sess)
    sess.close()

if __name__ == '__main__':
    run_cnnclassify(is_train=True)
