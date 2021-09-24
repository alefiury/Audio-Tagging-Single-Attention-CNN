import numpy as np
import torch
from sklearn import metrics


def model_accuracy(label, output):
    """
    Calculates the model accuracy considering its outputs.

    ----
    Args:
        output: Model output.
        label: True labels.

    Returns:
        Accuracy as a decimal.
    """

    pb = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
    _, top_class = pb.topk(1, dim=1)
    label = torch.argmax(label, dim=1)
    equals = label == top_class.view(-1)

    return torch.mean(equals.type(torch.FloatTensor).to('cuda'))


def model_f1_score(label, output):
    """
    Calculates the model f1-score considering its outputs.

    ----
    Args:
        output: Model output.
        label: True labels.

    Returns:
        F1 score as a decimal.
    """

    pb = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
    _, top_class = pb.topk(1, dim=1)
    label = torch.argmax(label, dim=1)

    return metrics.f1_score(label.detach().cpu().numpy(), top_class.detach().cpu().numpy(), average='macro')