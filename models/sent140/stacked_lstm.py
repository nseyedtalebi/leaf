import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import (
    DPGradientDescentGaussianOptimizer,
)

from tensorflow.contrib import rnn

from model import Model
from utils.language_utils import line_to_indices, get_word_emb_arr, val_to_vec


VOCAB_DIR = "sent140/embs.json"


class ClientModel(Model):
    def __init__(
        self,
        seed,
        lr,
        seq_len,
        num_classes,
        n_hidden,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        unroll_microbatches=False,
        emb_arr=None,
        ledger=None,
    ):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.vocab_size = len(vocab)
        if emb_arr:
            self.emb_arr = emb_arr
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = (num_microbatches,)
        self.unroll_microbatches = unroll_microbatches
        self.dp_delta = None
        self.dp_epsilon = None
        self.lr = lr
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable(
            "embedding", [self.vocab_size + 1, self.n_hidden], dtype=tf.float32
        )
        x = tf.cast(tf.nn.embedding_lookup(embedding, features), tf.float32)
        tf.print(x)
        labels = tf.placeholder(tf.float32, [None, self.num_classes])

        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)]
        )
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        fc1 = tf.layers.dense(inputs=outputs[:, -1, :], units=128)
        pred = tf.layers.dense(inputs=fc1, units=self.num_classes)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels)

        assert self.l2_norm_clip and self.noise_multiplier
        self.optimizer = DPGradientDescentGaussianOptimizer(
            self.l2_norm_clip,
            self.noise_multiplier,
            num_microbatches=1,
            ledger=None,
            unroll_microbatches=True,
            learning_rate=self.lr,
        )

        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            colocate_gradients_with_ops=True,
        )

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, self.indd, max_words) for e in x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        y_batch = [val_to_vec(self.num_classes, e) for e in y_batch]
        y_batch = np.array(y_batch)
        return y_batch
