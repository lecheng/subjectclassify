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
    EMBEDDING_ROOT = os.path.join(PROJECT_ROOT, 'models/embedding')
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
    threshold = 0.8
    learning_rate = 1e-3
    embedding_size = 200
    class_num = 1771
    vocab_size = 100000
    epochs = 48
    filters_num = 128
    features_each_filter = 1
    hidden_dim = 100
    text_length = 300
    batch_size = 128
    keep_prob = 0.8
    data_dir = BasicConfig.DATA_ROOT + '/'
    checkpoints_dir = BasicConfig.CHECKPOINTS_ROOT + '/'
    model_dir = BasicConfig.CHECKPOINTS_ROOT + '/'
    embedding_dir = BasicConfig.EMBEDDING_ROOT + '/'
