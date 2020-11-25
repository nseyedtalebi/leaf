from abc import abstractmethod

from tensorflow_privacy.privacy.optimizers.dp_optimizer import *

from model import Model


class DPGaussianModel(Model):
    def __init__(
        self,
        seed,
        lr,
        l2_norm_clip,
        noise_multiplier,
        optimizer=None,
        num_microbatches=None,
        unroll_microbatches=False,
    ):
        n, batch_size, noise_multiplier, epochs, delta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = num_microbatches
        self.unroll_microbatches = unroll_microbatches
        if optimizer is None:
            myopt = DPGradientDescentGaussianOptimizer(
                l2_norm_clip, noise_multiplier, num_microbatches, unroll_microbatches
            )
            super(DPModel, self).__init__(seed, lr, myopt)
        elif isinstance(
            optimizer,
            (
                AdagradOptimizer,
                AdamOptimizer,
                GradientDescentOptimizer,
                RMSPropOptimizer,
            ),
        ):
            DPGaussianOptimizer = make_gaussian_optimizer_class(optimizer)
            myopt = DPGaussianOptimizer(
                l2_norm_clip, noise_multiplier, num_microbatches, unroll_microbatches
            )
            super(DPModel, self).__init__(seed, lr)
        else:
            raise AttributeError(
                "Optimizer must be in tensorflow_privacy.privacy.optimizers for DP model"
            )
