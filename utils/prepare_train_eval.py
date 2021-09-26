import torch

import pandas as pd
from omegaconf import DictConfig

from trainer import train_model
from utils.evaluate import test_model
from utils.preload_data import prepare_data

def prepare_train(cfg : DictConfig,
                    fold: int,
                    metadata_path: str,
                    data_path: str,
                    output_path: str) -> None:
    """
    Loads metadata and prepares data to train the models.

    ----
    Args:
        cfg: A DictConfig given by hydra.
        fold: Fold identifier.
        metadata_path: Path to the metadata of the audio files.
        data_path: Path to the directory where the audio files are stored.
        output_path: Path to the directory where the model weights will be saved.
    """

    # Load csv metadata file
    df = pd.read_csv(metadata_path)
    # Filter train data by fold
    df_train = df[df['fold']!=fold]
    # Filter test data by fold
    df_val = df[df['fold']==fold]

    # Load train data from disk
    dataset_train = prepare_data(df=df_train,
                                    data_path=data_path,
                                    class_num=cfg.model.class_num,
                                    sample_rate=cfg.model.sample_rate,
                                    num_workers=cfg.train.num_workers)

    # Load validation data from disk
    dataset_val = prepare_data(df=df_val,
                                    data_path=data_path,
                                    class_num=cfg.model.class_num,
                                    sample_rate=cfg.model.sample_rate,
                                    num_workers=cfg.train.num_workers)

    # Convert preloaded data to torch tensor to use as input in the pytorch Dataloader
    dataset_train.set_format(type='torch', columns=['image', 'target', 'hot_target'])
    dataset_val.set_format(type='torch', columns=['image', 'target', 'hot_target'])

    # Drop last batch to make sure that the batch size has an even number of samples,
    # for all batches, so that the implementation of the mixup augmentation works correctly
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers
    )

    train_model(train_loader,
                    valid_loader,
                    output_path,
                    cfg=cfg)


def prepare_eval(cfg : DictConfig,
                    fold: int,
                    metadata_path: str,
                    data_path: str,
                    checkpoint_path: str) -> None:
    """
    Loads metadata and prepares data to evaluate the models.

    ----
    Args:
        cfg: A DictConfig given by hydra.
        fold: Fold identifier.
        metadata_path: Path to the metadata of the audio files.
        data_path: Path to the directory where the audio files are stored.
        checkpoint_path: Path to the base directory where the models weights were saved.
    """

    # Load csv metadata file
    df = pd.read_csv(metadata_path)

    # Filter test data by fold
    df_val = df[df['fold']==fold]

    # Preload test data
    dataset_val = prepare_data(df=df_val,
                                    data_path=data_path,
                                    class_num=cfg.model.class_num,
                                    sample_rate=cfg.model.sample_rate,
                                    num_workers=cfg.train.num_workers)

    # Convert to torch tensor
    dataset_val.set_format(type='torch', columns=['image', 'target', 'hot_target'])

    valid_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers
    )

    preds, labels = test_model(valid_loader, checkpoint_path, cfg)

    return preds, labels