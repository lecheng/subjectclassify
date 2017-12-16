import os
import tensorflow as tf

class BasicConfig(object):
    """
    basic configuration for the project
    """
    # PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'/')
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    MODEL_ROOT = os.path.join(PROJECT_ROOT, 'models')
    CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_LEVLE = 2
    # LOG_LEVEL:
    # |-- 0 - Only record error messages
    # |-- 1 - Record errors and warnings
    # |-- 2 - All available messages are recorded

class TCCNNConfig(object):
    """
    configuration for text classification cnn models
    """
    k = 2
    threshold = 0.6
    learning_rate = 1e-3
    embedding_size = 1000
    class_num = 1771
    vocab_size = 100000
    epochs = 24
    filters_num = 256
    features_each_filter = 10
    hidden_dim = 128
    text_length = 300
    batch_size = 128
    keep_prob = 1.0
    data_dir = BasicConfig.DATA_ROOT + '/'
    checkpoints_dir = BasicConfig.CHECKPOINTS_ROOT + '/'
    model_dir = BasicConfig.CHECKPOINTS_ROOT + '/'