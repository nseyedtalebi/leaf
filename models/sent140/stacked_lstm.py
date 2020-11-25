import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.compat.v1.nn import rnn_cell as rnn

from model import Model
from utils.language_utils import line_to_indices, get_word_emb_arr, val_to_vec
from tensorflow_privacy.privacy.optimizers.dp_optimizer import (
    DPGradientDescentGaussianOptimizer,
)

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
        dp_opt = DPGradientDescentGaussianOptimizer(
            l2_norm_clip,
            noise_multiplier,
            num_microbatches=None,
            ledger=None,
            unroll_microbatches=False,
            learning_rate=lr,
        )
        super(ClientModel, self).__init__(seed, lr, dp_opt)

    def create_model(self):
        features = tf.compat.v1.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.compat.v1.get_variable(
            "embedding",
            [self.vocab_size + 1, self.n_hidden],
            dtype=tf.float32,
            use_resource=False,
        )
        x = tf.cast(tf.nn.embedding_lookup(params=embedding, ids=features), tf.float32)
        labels = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes])

        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicRNNCell(self.n_hidden) for _ in range(2)]
        )
        # stacked_lstm = tf.keras.layers.RNN(rnn_cells)
        # stacked_lstm = CuDNNLSTM(2)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        # outputs = stacked_lstm(x)
        # outputs = stacked_lstm(x)
        fc1 = tf.compat.v1.layers.dense(inputs=outputs[:, -1, :], units=128)
        # fc1 = tf.compat.v1.layers.dense(inputs=outputs, units=128)
        pred = tf.compat.v1.layers.dense(inputs=fc1, units=self.num_classes)

        loss = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=pred, labels=labels
            )
        )
        train_op = self.optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step()
        )

        correct_pred = tf.equal(
            tf.argmax(input=pred, axis=1), tf.argmax(input=labels, axis=1)
        )
        eval_metric_ops = tf.math.count_nonzero(correct_pred)

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

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        noise = self.optimizer.noise_multiplier
        self.dp_delta = 1 / len(data["y"])  # using this for N
        self.dp_epsilon = None
        for epoch_num in range(num_epochs):
            self.run_epoch(data, batch_size)
            # Add DP stuff
            # if isinstance(self.optimizer, DPGradientDescentGaussianOptimizer):
            self.dp_epsilon = compute_dp_sgd_privacy(
                len(data["y"]), batch_size, noise, epoch_num, dp_delta
            )
            #    )
        update = self.get_params()
        comp = num_epochs * (len(data["y"]) // batch_size) * batch_size * self.flops
        return comp, update
