from typing import Callable, Any

import numpy as np

from optuna.samplers import TPESampler

ESP = 1e-6


def cost_decay(x: int) -> float:
    # sigmoid x
    return 1 / (1 + np.exp(-x))


class FrugalTPESampler(TPESampler):
    def __init__(
        self,
        cost_estimator: Callable[[dict[str, Any]], float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_estimator = cost_estimator
        self.cost_decay_cnt = 6

    def _compute_acquisition_func(self, samples, mpe_below, mpe_above):
        log_likelihood_diff = super()._compute_acquisition_func(
            samples, mpe_below, mpe_above
        )

        for i in range(log_likelihood_diff.size):
            params = {k: v[i].item() for k, v in samples.items()}
            cost = self.cost_estimator(params) + ESP
            cost = cost ** cost_decay(self.cost_decay_cnt)  # to moderate cost effect
            log_likelihood_diff[i] -= np.log(cost)
        self.cost_decay_cnt = max(-10, self.cost_decay_cnt - 1)
        return log_likelihood_diff
