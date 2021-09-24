import torch
import torch.nn as nn

class BCELossModified(nn.Module):
    """
    Modified version of Binary Cross Entropy.
    Deal with cases where the probabilities are
    infinite (inf) or unknown (nan).
    Clip the output to be in the range [0,1].
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)

        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        input_ = input_.clamp(0, 1)

        return self.bce(input_, target)