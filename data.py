import pandas as pd
import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr
import re, io, os

class Subject:
    def __init__(self):
        self.class_num = 783
        self.vocab_size = 100678

    def get_label_dict(self, path='data/subject_node.txt'):
        label_dict = []
        with open(path,'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line[1:-1].split(',')[0].replace("\"","")
                label_dict += [label]
        cat_to_id = dict(zip(label_dict, range(len(label_dict))))
        return label_dict, cat_to_id

    def label_count(self, path, subject = None):
        dataobj = self.get_valid_data(path, subject)
        obj_labels = list(dataobj['labels'])
        labels = []
        for label in obj_labels:
            label = label[1:-1].replace('\"', '').split(',')
            labels += label
        label_count = Counter(labels)
        cat_to_id = dict(zip(label_count, range(len(label_count))))
        print('total label number: {0}'.format(len(label_count)))
        return label_count, cat_to_id

    def get_valid_data(self, path, subject = None):
        data = pd.read_csv(path, encoding='utf-8')
        mask = (data['labels'].str.len() > 2) & (data['abstract'].str.len() > 20)
        data = data.loc[mask]
        if subject is not None:
            data = data[data['subject'].isin(subject)]
        return data

    def remove_html_tag(self, abstract):
        return re.sub('<.*?>','',abstract)

    def save_valid_data(self, data):
        print len(data)
        train_path = 'data/train.txt'
        test_path = 'data/test.txt'
        val_path = 'data/val.txt'
        f_train = io.open(train_path, 'w')
        f_test = io.open(test_path, 'w')
        f_val = io.open(val_path, 'w')
        abstracts = list(data['abstract'])
        labels = list(data['labels'])
        for i in range(0, len(data)):
            abstract = abstracts[i]
            abstract = self.remove_html_tag(abstract)
            label = labels[i]
            label = label[1:-1].replace("\"", "")
            if i < 1000:
                f_test.writelines(label + '\t' + abstract + '\n')
            elif i < 1000:
                f_val.writelines(label + '\t' + abstract + '\n')
            else:
                f_train.writelines(label + '\t' + abstract + '\n')
        f_train.close()
        f_test.close()
        f_val.close()

    def build_vocal(self, data, vocab_size=100000):
        data = list(data['abstract'])
        print 'total valid text num: {0}'.format(len(data))
        all_data = []
        for content in data:
            if content:
                content = self.remove_html_tag(content)
                content = re.sub('[().,;:]', '', content)
                words = content.split(' ')
                all_data.extend(words)
        print 'total words in abstracts: {0}'.format(len(all_data))

        counter = Counter(all_data)
        print 'total unique words in abstrats: {0}'.format(len(counter))
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # add a tag <PAD> to make all the text the same length
        words = ['<PAD>'] + list(words)
        print(len(words))
        # vocab = []
        # for word in words:
        #     if re.match('.*<.*>.*',word) or ('=' in word):
        #         continue
        #     vocab += [word]
        # print(len(vocab))
        io.open('data/vocab.txt', 'w').write('\n'.join(words))


    def read_file(self, filename):
        """
        get label and content from file
        :param filename: (String) file name
        :return: (List of List of String)content list, (List of String)label list
        """
        contents = []
        labels = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                try:
                    label, content = line.strip().split('\t')
                    content = re.sub('[().,;:]', '', content)
                    contents.append(content.split(' '))
                    labels.append(label.split(','))
                except:
                    pass
        return contents, labels

    def file_to_ids(self, filename, word_to_id, cat_to_id, max_length=100):
        """
        transfer word and label to id list
        :param filename: (String) file name
        :param word_to_id: (Dict) word to id dictionary
        :param max_length: (Int) max length of file
        :return: (List of List of Int)id list, (List of Int)label id list
        """
        contents, labels = self.read_file(filename)

        data_id = []
        label_vectors = np.zeros((len(labels), len(cat_to_id)))
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            for x in labels[i]:
                if x in cat_to_id:
                    label_vectors[i][cat_to_id[x]] = 1

        # limited text into fixed length
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        return x_pad, label_vectors

    def read_vocab(self, filename):
        """
        :param filename: file name
        :return: words vocaburary and word to id dictionary
        """
        words = list(map(lambda line: line.strip(),
                    io.open(filename, 'r', encoding='utf-8').readlines()))
        word_to_id = dict(zip(words, range(len(words))))

        return words, word_to_id

    def process_file(self, data_path='data/', seq_length=100, subject = None):
        """
        :param data_path: data file path
        :param seq_length: max length of file
        :return: 
        """
        words, word_to_id = self.read_vocab(os.path.join(data_path,
                                                         'vocab.txt'))
        # _, cat_to_id = self.get_label_dict(os.path.join(data_path,
        #                                                  'subject_node.txt'))
        _, cat_to_id = self.label_count(os.path.join(data_path,
                                                         'paper.csv'), subject)
        x_train, y_train = self.file_to_ids(os.path.join(data_path,
                                                         'train.txt'), word_to_id, cat_to_id, seq_length)
        x_test, y_test = self.file_to_ids(os.path.join(data_path,
                                                       'test.txt'), word_to_id, cat_to_id, seq_length)
        x_val, y_val = self.file_to_ids(os.path.join(data_path,
                                                     'val.txt'), word_to_id, cat_to_id, seq_length)
        return x_train, y_train, x_test, y_test, x_val, y_val

class Eurlex:
    def __init__(self):
        self.class_num = 3993
        self.vocab_size = 5001

    def extract_label_and_features(self, filename, max_length=100):
        lines = io.open(filename, 'r', encoding='utf-8').readlines()
        feature_matrix = []
        label_matrix = []
        count = 1
        for line in lines:
            # print count
            count += 1
            parts = line.split(' ')
            labels = parts[0].split(',')
            # print labels
            if len(labels[0])>0:
                features = parts[1:]
                label_vector = np.zeros(self.class_num)
                for label in labels:
                    label_vector[int(label)] = 1
                f_list = []
                for feature in features:
                    key, value = feature.split(':')
                    f_list += [int(key)+1]
                feature_matrix.append(f_list)
                label_matrix.append(label_vector)
        x_pad = kr.preprocessing.sequence.pad_sequences(feature_matrix, max_length)
        return x_pad, label_matrix

    def process_file(self, data_path='data/Eurlex', seq_length=100):
        training_path = data_path + '/eurlex_train.txt'
        test_path = data_path + '/eurlex_test.txt'
        x_train, y_train = self.extract_label_and_features(training_path, seq_length)
        x_test, y_test = self.extract_label_and_features(test_path, seq_length)
        x_val = x_test[:1000]
        y_val = y_test[:1000]
        x_test = x_test[1000:]
        y_test = y_test[1000:]
        return x_train, y_train, x_test, y_test, x_val, y_val

if __name__ == '__main__':
    obj = Subject()
    biology_subjects = ['Biochemistry', 'Biological techniques', 'Biophysics', 'Biotechnology', 'Cancer',
                        'Cell biology', 'Chemical biology', 'Computational biology and bioinformatics',
                        'Developmental biology', 'Drug discovery', 'Ecology', 'Evolution', 'Genetics', 'Immunology',
                        'Microbiology', 'Molecular biology', 'Neuroscience', 'Physiology', 'Plant sciences',
                        'Psychology', 'Stem cells', 'Structural biology', 'Systems biology', 'Zoology']

    data = obj.get_valid_data('data/paper.csv',['Oncology'])
    obj.build_vocal(data)
    obj.save_valid_data(data)
    # dataObj = Eurlex()
    # x_train, y_train, x_test, y_test, x_val, y_val = dataObj.process_file()
    # print x_train
    # print y_train
    # print np.array(x_train).shape
    # print np.array(y_train).shape
