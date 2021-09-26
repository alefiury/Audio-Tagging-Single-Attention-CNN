"""
    Mixup implementation based on:
        https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/750c318c0fcf089bd430f4d58e69451eec55f0a9/utils/utilities.py
        https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/tools/mixup.py
"""

import torch
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    """
    Mixup x of even indexes (0, 2, 4, ...)
    with x of odd indexes (1, 3, 5, ...).

    ----
    Args:
        x: (batch_size * 2, ...), batch_size must be even.
        mixup_lambda: (batch_size * 2,).

    Returns:
        output shape: (batch_size, ...)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)

    return out


class Mixup(object):
    def __init__(self, mixup_alpha):
        """
        Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha

    def get_lambda(self, batch_size):
        """
        Get mixup random coefficients.

        ----
        Args:
            batch_size: int

        Returns:
            mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32)).to(device)