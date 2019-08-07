# Original code
#
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# paper reference: https://arxiv.org/pdf/1301.3781.pdf
#



import random
import heapq
import numpy as np

from collections import defaultdict

from six.moves import xrange
from six import iteritems, itervalues

from scipy.special import expit


class Vocab(object):

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.count < other.count


class Word2Vec(object):

    """
    model_name -> 'CBOW' or 'SKIP_GRAM'
    loss_type -> 'general' or 'negative'
    sentences -> [['token11', 'token12', 'token13',..], ['token21', 'token22', ...], ..., ['token31', 'token32', 'token33']]
    window -> how long do you want to consider?
    hidden_dim -> dimension of hidden layer
    epoch_num -> training epoch
    learning_rate -> gradient descent learning rate
    subsample_threshold -> 'threshold for subsampling frequent words'
    """

    def __init__(self, model_name, loss_type, sentences, window, hidden_dim=512, epoch_num=3, learning_rate=0.001,
                 subsample_threshold=pow(10,-5)):
        self.index2word = []
        self.vocab_dict = {}

        self.model_name = model_name
        self.loss_type = loss_type
        self.window = window
        self.hidden_dim = hidden_dim
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.subsample_threshold = subsample_threshold

        self.preprocess(sentences)                                 # create count-based index, dictionary, huffman tree
        self.build_graph()                                         # create layer(projection layer, hidden layer, output layer)
        self.train(sentences)                                      # train graph with sentences



    def preprocess(self, sentences):
        vocab = defaultdict(int)
        for sen_num, sentence in enumerate(sentences):
            for word in sentence:
                vocab[word] += 1

        self.total_words = sum(itervalues(vocab))
        self.vocab_list = vocab

        for word, count in iteritems(self.vocab_list):
            subsample_prob = 1 - np.sqrt(self.subsample_threshold / (float(count) / float(self.total_words)))
            self.vocab_dict[word] = Vocab(count=count, index=len(self.index2word), sample_prob=subsample_prob)
            self.index2word.append(word)

        self.index2word.sort(key=lambda word_: self.vocab_dict[word_].count, reverse=True)

        for index, word in enumerate(self.index2word):
            self.vocab_dict[word].index = index

        huff_tree = list(itervalues(self.vocab_dict))
        heapq.heapify(huff_tree)

        for i in xrange(len(self.vocab_dict) - 1):
            min1, min2 = heapq.heappop(huff_tree), heapq.heappop(huff_tree)
            heapq.heappush(huff_tree, Vocab(count=min1.count + min2.count, index=i+len(self.vocab_dict), left=min1, right=min2))

        stack = [(huff_tree[0], [])]
        max_code_len = 0

        while stack:
            node, binary_code = stack.pop()
            if node.index < len(self.vocab_dict):
                node.binary_code = binary_code
                max_code_len = max(max_code_len, len(binary_code))

            else:
                stack.append((node.left, np.array(list(binary_code) + [0], dtype=np.uint8)))
                stack.append((node.right, np.array(list(binary_code) + [1], dtype=np.uint8)))

        self.max_code_len = max_code_len


    def build_graph(self):
        self.projection_layer = np.empty((len(self.vocab_dict), self.hidden_dim), dtype=np.float)   #(vocab_size, hidden_dim)
        self.hidden_layer = np.tanh(np.empty((self.hidden_dim, self.hidden_dim), dtype=np.float))   #(hidden_dim, hidden_dim) + tanh()
        self.output_layer = np.empty((self.max_code_len, self.hidden_dim), dtype=np.float)          #(hidden_dim, binary_code_len)


    def train(self, sentences):
        for epoch in xrange(self.epoch_num):
            for sentence in sentences:
                word_vocabs = [self.vocab_dict[word] for word in sentence if word in self.vocab_dict
                               and self.vocab_dict[word].sample_prob < random.random()]  # sub_samping freqeunt word
                for pos, word in enumerate(word_vocabs):
                    current = max(0, pos - self.window)
                    word2_indices = [word2.index for pos2, word2 in enumerate(word_vocabs[current:(pos + self.window + 1)], current)
                                     if (word2 is not None and pos2 != pos)]
                    if self.model_name == 'CBOW':
                        self.train_per_epoch_CBOW(word2_indices, self.index2word[word.index])
                    elif self.model_name == 'SKIP_GRAM':
                        self.train_per_epoch_SKIP_GRAM(word2_indices, self.index2word[word.index])
                    else:
                        raise NotImplementedError




    def train_per_epoch_CBOW(self, word2_indices, target_word):
        predict_word = self.vocab_dict[target_word]

        input_vector_list = [self.projection_layer[word_index] for word_index in word2_indices]
        input_vector = np.mean(input_vector_list, axis=0)
        tanh_output = np.dot(input_vector, self.hidden_layer)
        code_prediction = expit(np.dot(tanh_output, self.output_layer.T))
        loss_gradient = -np.abs(predict_word.binary_code - code_prediction) * self.learning_rate
        #loss_gradient = -(predict_word.binary_code + code_prediction) * self.learning_rate
        self.learn_vector(input_vector, input_vector_list, loss_gradient, tanh_output, model='CBOW')

        if self.loss_type == 'negative':
            negative_word = self.vocab_dict[random.randrange(len(self.vocab_dict))]
            loss_gradient = np.abs(negative_word.binary_code - code_prediction) * self.learning_rate
            #loss_gradient = (negative_word.binary_code - code_prediction) * self.learning_rate
            self.learn_vector(input_vector, input_vector_list, loss_gradient, tanh_output, model='CBOW')



    def train_per_epoch_SKIP_GRAM(self, word2_indices, target_word):
        predict_word = self.vocab_dict[target_word]

        for word_index in word2_indices:
            input_vector = self.projection_layer[word_index]
            tanh_output = np.dot(input_vector, self.hidden_layer)
            code_prediction = expit(np.dot(tanh_output, self.output_layer.T))
            loss_gradient = -np.abs(predict_word.binary_code - code_prediction) * self.learning_rate
            #loss_gradient = -(predict_word.binary_code + code_prediction) * self.learning_rate
            self.learn_vector(input_vector, None, loss_gradient, tanh_output, model='SKIP_GRAM')

            if self.loss_type == 'negative':
                negative_word = self.vocab_dict[random.randrange(len(self.vocab_dict))]
                loss_gradient = np.abs(negative_word.binary_code - code_prediction) * self.learning_rate
                #loss_gradient = (negative_word.binary_code - code_prediction) * self.learning_rate
                self.learn_vector(input_vector, None, loss_gradient, tanh_output, model='SKIP_GRAM')


    def learn_vector(self, input_vector, input_vector_list, loss_gradient, tanh_output, model='SKIP_GRAM'):
        # tanh_output -> (1 - pow(tanh_output,2)) [hidden_state]
        input_vector_step = np.dot(np.dot(loss_gradient, self.output_layer), self.hidden_layer.T)
        hidden_layer_step = np.outer(input_vector, np.dot(loss_gradient, self.output_layer)) * (1 - pow(self.hidden_layer,2))
        output_layer_step = np.outer(loss_gradient, tanh_output)
        self.hidden_layer += hidden_layer_step
        self.output_layer += output_layer_step

        if model == 'SKIP_GRAM':
            input_vector += input_vector_step

        elif model == 'CBOW':
            for input_vector_ in input_vector_list:
                input_vector_ += input_vector_step

        else:
            raise NotImplementedError


    def most_similar(self, target_words, top_most=11):
        #['akc', 'dfe', 'bec', ..., 'wbp']
        def normalize(array1, array2):
            array_1 = array1 /np.linalg.norm(array1)
            array_2 = array2 / np.linalg.norm(array2)
            return np.dot(array_1, array_2)

        def get_cosine_similarity(cosine_similarity):
            word_list = []
            cosine_similarity[cosine_similarity.index(max(cosine_similarity))] = 0 #remove same word
            for i in xrange(top_most):
                index = cosine_similarity.index(max(cosine_similarity))
                cosine_similarity[index] = 0
                word_list.append(self.index2word[index])
            return word_list

        similar_words_list = []
        for target_word in target_words:
            try:
                vocab = self.vocab_dict[target_word]
            except KeyError:
                print('This word does not exist!')
                continue

            target_word_vector = self.projection_layer[vocab.index]
            cosine_similarity = [normalize(target_word_vector, embedding_vector) for embedding_vector in self.projection_layer] #[vocab_size]
            print(self.index2word)
            print(cosine_similarity)
            similar_words = get_cosine_similarity(cosine_similarity)
            similar_words_list.append(similar_words)

        return similar_words_list































#projection_to_hidden = np.dot(l1, self.hidden_layer)
#hidden_to_output = np.dot(projection_to_hidden, self.output_layer.T)
#code_prediction = expit(hidden_to_output)

#cosine_similarity = np.dot(target_word_vector, self.projection_layer.T)     #[vocab_size]





