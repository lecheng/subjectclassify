import os
import datetime
import time

from config import BasicConfig
from warnings import warn

class Logger(object):
    def __init__(self, file_path=None):
        '''
        :param file_path: (String) path to log file
        '''
        if file_path is None:
            self.log_file = open(os.path.join(BasicConfig.PROJECT_ROOT, 'info.log'), 'a+')
        else:
            try:
                self.log_file = open(file_path, 'a+')
            except IOError as e:
                print('Log file does not exists. Further logs would only be printed on screen and would not be saved.')

    def info(self, msg):
        '''
        :param msg: (String) message to save
        :return: None
        '''
        t = (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        if BasicConfig.LOG_LEVLE >= 2:
            print('[INFO] ' + msg)
            self.log_file.write(str(t) + ' [INFO] ' + msg + '\n')

    def warning(self, msg):
        '''
        :param msg: (String) warning message
        :return: None
        '''
        t = (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        if BasicConfig.LOG_LEVLE >= 1:
            warn('[WARNING] ' + msg)
            self.log_file.write(str(t) + ' [WARNING] ' + msg + '\n')

    def error(self, msg):
        '''
        :param msg: (String) error message
        :return: None
        '''
        t = (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        warn('[ERROR]' + msg)
        self.log_file.write(str(t) + '[ERROR] ' + msg + '\n')

class EmptyLogger(Logger):
    '''
    EmptyLogger is a shell for class Logger, only designed for debugging.
    EmptyLogger prints out all debugging output to console.
    '''

    def __init__(self, file_path=None):
        pass

    def info(self, msg):
        print('[INFO] ' + msg)

    def warning(self, msg):
        print('[WARNING] ' + msg)

    def error(self, msg):
        warn('[ERROR] ' + msg)