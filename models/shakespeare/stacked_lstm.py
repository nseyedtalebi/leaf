import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        features = tf.compat.v1.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.compat.v1.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(params=embedding, ids=features)
        labels = tf.compat.v1.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.compat.v1.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(input=labels, axis=1))
        eval_metric_ops = tf.math.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch
