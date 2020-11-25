from abc import abstractmethod

from tensorflow_privacy.privacy.optimizers.dp_optimizer import *

from model import Model


class DPModel(Model):
    def __init__(
        self,
        seed,
        lr,
        dp_sum_query,
        optimizer=None,
        num_microbatches=None,
        unroll_microbatches=False,
    ):
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = num_microbatches
        self.unroll_microbatches = unroll_microbatches
        if optimizer is None:
            myopt = DPGradientDescentOptimizer(
                dp_sum_query, num_microbatches, unroll_microbatches
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
            DPOptimizer = make_optimizer_class(optimizer)
            myopt = DPOptimizer(dp_sum_query, num_microbatches, unroll_microbatches)
            super(DPModel, self).__init__(seed, lr)
        else:
            raise AttributeError(
                "Optimizer must be in tensorflow_privacy.privacy.optimizers for DP model"
            )
