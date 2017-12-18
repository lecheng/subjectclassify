import os
import tensorflow as tf
import numpy as np
from logger import Logger

biology_subjects = ['Biochemistry','Biological techniques','Biophysics','Biotechnology','Cancer','Cell biology','Chemical biology','Computational biology and bioinformatics','Developmental biology','Drug discovery','Ecology','Evolution','Genetics','Immunology','Microbiology','Molecular biology','Neuroscience','Physiology','Plant sciences','Psychology','Stem cells','Structural biology','Systems biology','Zoology']


class TextClassificationCNN(object):

    def __init__(self, config, dataObj, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()
        self.config.class_num = dataObj.class_num
        self.KERNEL_SIZE = [1,2,3,4,5]
        self.config.vocab_size = dataObj.vocab_size
        self.input_x = tf.placeholder(tf.int32, [None, self.config.text_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.class_num], name='input_y')
        self.thre = tf.placeholder(tf.float32, name='threshold')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_train = True
        self._data_process(dataObj)
        self._build()

    def _data_process(self, dataObj):
        self.x_train, self.y_train, self.x_test, self.y_test,\
            self.x_val, self.y_val = dataObj.process_file(self.config.data_dir, self.config.text_length, biology_subjects)

    def _build(self):
        with tf.device('/cpu:0'):
            embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_size],
                                -1.0, 1.0), name='embedding')
            embeded = tf.nn.embedding_lookup(embedding, self.input_x)
        self.logger.info('embeded shape {0}'.format(embeded.shape))
        maxpooling_outputs = []
        for kernel_size in self.KERNEL_SIZE:
            conv_name = 'conv'+str(kernel_size)
            conv = tf.layers.conv1d(embeded, self.config.filters_num, kernel_size, padding='same', activation=tf.nn.tanh, name=conv_name)

            pool_size = self.config.text_length / self.config.features_each_filter
            maxpooling = tf.layers.max_pooling1d(conv, pool_size, pool_size)
            # maxpooling = tf.reduce_max(conv, reduction_indices=[1])
            maxpooling_outputs.append(maxpooling)
        total_features = self.config.features_each_filter * len(self.KERNEL_SIZE) * self.config.filters_num
        # gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        self.logger.info('conv shape {0}'.format(conv.shape))
        self.logger.info('maxpooling shape {0}'.format(maxpooling.shape))
        gmp = tf.transpose(tf.concat(maxpooling_outputs, 2), perm=[0,2,1])
        # gmp = tf.concat(maxpooling_outputs, 1)
        self.logger.info('gmp shape {0}'.format(gmp.shape))
        gmp = tf.reshape(gmp, [-1, total_features])
        self.logger.info('gmp shape {0}'.format(gmp.shape))

        # fc1 = tf.contrib.layers.fully_connected(gmp, self.config.hidden_dim)
        fc1 = tf.contrib.layers.fully_connected(gmp, self.config.class_num, activation_fn=None)
        if self.is_train:
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
        self.logger.info('fc1 shape {0}'.format(fc1.shape))
        self.logits = fc1
        # self.logits = tf.contrib.layers.fully_connected(fc1, self.config.class_num, activation_fn=None)
        self.logger.info('logits shape {0}'.format(self.logits.shape))
        # self.predict_y = tf.argmax(self.logits,1,output_type=tf.int32)
        # self.predict_y = tf.round(tf.nn.sigmoid(self.logits))
        self.k = self.config.k
        topk, idx = tf.nn.top_k(self.logits,k=self.k)
        idx, _ = tf.nn.top_k(-idx,k=self.k)
        # self.logger.info('idx shape {0}'.format(idx.shape))
        shape = tf.convert_to_tensor([self.config.batch_size, self.config.class_num], tf.int32)
        #
        indices = tf.stack([tf.tile(tf.range(0, self.config.batch_size)[..., tf.newaxis], [1, self.k]), -idx], axis=2)
        indices = tf.reshape(tf.squeeze(indices), [-1, 2])

        self.predict_y = tf.sparse_to_dense(indices, shape, 1.0, 0.0)
        # self.predict_y = tf.cast(self.predict_y, tf.float32)
        # self.predict_y = tf.cast(tf.greater(tf.nn.sigmoid(self.logits), self.thre), tf.float32)
        self.logger.info('predict_y shape {0}'.format(self.predict_y))
        # output_onehot = tf.one_hot(self.input_y, self.config.class_num)
        # self.logger.info('one hot shape {0}'.format(output_onehot.shape))
        self.correct_pred = tf.equal(self.predict_y, tf.round(tf.cast(self.input_y, tf.float32)))
        sum1 = tf.reduce_sum(self.predict_y,1)
        self.correct_positive_pred = tf.multiply(self.predict_y, tf.cast(self.input_y, tf.float32))
        sum2 = tf.reduce_sum(self.correct_positive_pred,1)
        all_pred_labels_true = tf.equal(sum1, sum2)
        self.all_correct_accuracy = tf.reduce_mean(tf.cast(all_pred_labels_true, tf.float32))
        self.recall = tf.divide(tf.reduce_sum(self.correct_positive_pred), tf.reduce_sum(tf.cast(self.input_y, tf.float32)))
        self.precision = tf.divide(tf.reduce_sum(self.correct_positive_pred), tf.reduce_sum(self.predict_y))

        # self.loss = tf.nn.weighted_cross_entropy_with_logits(logits=self.logits, targets=tf.cast(self.input_y, tf.float32), pos_weight=1)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.input_y, tf.float32))
        self.total_loss = tf.reduce_mean(self.loss)
        self.pred_one_num = tf.reduce_sum(self.predict_y)
        self.total_label_num = tf.reduce_sum(tf.cast(self.input_y, tf.float32))
        self.true_positive = tf.reduce_sum(self.correct_positive_pred)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.total_loss)
        # correct_pred = tf.equal(self.input_y, self.predict_y)
        # self.logger.info('input_y shape {0}'.format(self.input_y.shape))
        # self.logger.info('predict_y shape {0}'.format(self.predict_y.shape))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, sess):
        self.is_train = True
        if not os.path.exists(os.path.dirname(self.config.checkpoints_dir)):
            os.mkdir(os.path.dirname(self.config.checkpoints_dir))
        if not os.path.exists(self.config.checkpoints_dir):
            os.mkdir(self.config.checkpoints_dir)

        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            self.logger.info('restore from checkpoint {0}'.format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        self.logger.info('start training...')
        try:
            for epoch in range(start_epoch, self.config.epochs):
                iterations = len(self.x_train) // self.config.batch_size
                self.logger.info('total iterations: {0}'.format(iterations))
                random_indices = np.random.permutation(np.arange(len(self.x_train)))
                shuffled_x_train = np.array(self.x_train)[random_indices]
                shuffled_y_train = np.array(self.y_train)[random_indices]
                for i in range(iterations):
                    start_index = i * self.config.batch_size
                    end_index = (i + 1) * self.config.batch_size
                    x_train = shuffled_x_train[start_index:end_index]
                    y_train = shuffled_y_train[start_index:end_index]
                    loss, _, accuracy, a, b, c = sess.run([
                        self.total_loss,
                        self.train_op,
                        self.all_correct_accuracy,
                        self.precision,
                        self.recall,
                        self.total_label_num,
                    ], feed_dict = {self.input_x: x_train, self.input_y: y_train,
                                    self.keep_prob: self.config.keep_prob, self.thre: self.config.threshold})
                    self.logger.info('Epoch: {0}, iteration: {1}, training loss: {2}, training accuracy: {3}, precision: {4}, recall: {5}, total label num: {6}, '
                                     .format(epoch, i, loss, accuracy, a, b, c))
                self.logger.info('start evaluating epoch {0}...'.format(epoch))
                self.evaluate(sess)
                if (epoch+1) % 6 == 0:
                    saver.save(sess, self.config.model_dir, global_step= epoch)
        except KeyboardInterrupt:
            self.logger.error('Interrupt manually, try saving checkpoint for now...')
            print self.config.model_dir
            saver.save(sess, self.config.model_dir, global_step = epoch)
            self.logger.info('Last epoch were saved, next time will start from epoch {0}.'.format(epoch))
            self.logger.info('start evaluating epoch {0}...'.format(epoch))
            self.evaluate(sess)

    def evaluate(self, sess):
        self.is_train = False
        iterations = len(self.x_val) // self.config.batch_size
        self.logger.info('total iterations: {0}'.format(iterations))
        random_indices = np.random.permutation(np.arange(len(self.x_val)))
        shuffled_x_val = np.array(self.x_val)[random_indices]
        shuffled_y_val = np.array(self.y_val)[random_indices]
        total_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for i in range(iterations):
            start_index = i * self.config.batch_size
            end_index = (i + 1) * self.config.batch_size
            x_val = shuffled_x_val[start_index:end_index]
            y_val = shuffled_y_val[start_index:end_index]
            loss, accuracy, recall, precision = sess.run([self.total_loss, self.all_correct_accuracy, self.recall, self.precision],
                            feed_dict={
                                self.input_x: x_val,
                                self.input_y: y_val,
                                self.thre: self.config.threshold,
                                self.keep_prob: self.config.keep_prob
                            })
            total_loss += loss
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            self.logger.info('iteration: {0}, evaluation loss: {1}, accuracy: {2}, precision: {3}, recall: {4}'
                             .format(i, loss, accuracy, precision, recall))
        self.logger.info('average evaluation loss: {0}, average accuracy: {1}, average precision: {2}, average recall: {3}'
                         .format(total_loss/iterations, total_accuracy/iterations, total_precision/iterations, total_recall/iterations))

    def predict(self,sess, input_x):
        self.is_train = False
        self.config.batch_size = 1
        self._build()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        predict_y = sess.run(self.predict_y,feed_dict={self.input_x: input_x, self.thre: self.config.threshold})
        self.logger.info('prediction finished...')
        return predict_y

