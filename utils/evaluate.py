import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from omegaconf import DictConfig

from model import Cnn_Single_Att, Wavegram_Logmel_Cnn_Single_Att

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(
    valid_loader: torch.utils.data.DataLoader,
    checkpoint_path: str,
    cfg : DictConfig
) -> None:
    """
    Predicts new data.

    ----
    Args:
        valid_loader: Dataloader for the evaluation data.

        checkpoint_path: Path to saved weights.

        fold: Current validation fold.
    """

    preds = []
    labels = []

    if cfg.train.use_wavgram_logmel:
        model = Wavegram_Logmel_Cnn_Single_Att(**cfg.model)
    else:
        model = Cnn_Single_Att(**cfg.model)
    model.to(device)

    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    with torch.no_grad():
        model.eval()

        for val_sample in tqdm(valid_loader):

            val_audio, val_label = val_sample['image'].to(device), val_sample['target']
            output = model(val_audio)

            preds_hot = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
            preds_arg = torch.argmax(preds_hot, dim=1).cpu().detach().numpy()

            labels_arg = val_label.cpu().detach().numpy()

            preds.append(preds_arg)
            labels.append(labels_arg)

    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    return preds, labels