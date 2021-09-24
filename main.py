import os
import glob
import logging

import numpy as np
import torch
import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from sklearn.metrics import (accuracy_score, f1_score,
                                precision_score, recall_score)

from utils.prepare_train_eval import prepare_train, prepare_eval

device = ('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login()

# Logger
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="default")
def main(cfg : DictConfig) -> None:

    # Get current directory path to deal with Hydra logging path
    original_cwd = get_original_cwd()
    # Creates a directory to save wandb logs outside of Hydra logging path
    os.makedirs(os.path.join(original_cwd, 'wandb'), exist_ok=True)


    if cfg.command == 'train':
        wandb_config = {**cfg.train,
                    **cfg.model,
                    **cfg.data}

        for fold in cfg.train.folds[4:]:
            log.info(f"Training fold: {fold} -> {len(cfg.train.folds)}... ")
            # Start a new run on wandb
            run = wandb.init(project=os.path.basename(cfg.data.output_dir),
                                config=wandb_config,
                                dir=os.path.join(original_cwd, 'wandb'),
                                reinit=True)

            # Change run name to facilitate statistics traceability
            wandb.run.name = 'fold '+str(fold)
            wandb.run.save()

            prepare_train(cfg=cfg,
                            fold=fold,
                            metadata_path=os.path.join(original_cwd, cfg.data.metadata_path),
                            data_path=os.path.join(original_cwd, cfg.data.data_path),
                            output_path=os.path.join(original_cwd, cfg.data.output_dir, 'fold_'+str(fold)))

            # Finish run
            run.finish()


    elif cfg.command == 'test':
        accs = []
        f1_scores = []
        precisions = []
        recalls = []
        for fold in cfg.train.folds:
            log.info(f"Testing fold: {fold} -> {len(cfg.train.folds)}... ")

            # Get saved weights paths
            checkpoint_path = glob.glob(os.path.join(original_cwd, cfg.data.output_dir, 'fold_'+str(fold), '*.pth'))[0]

            preds, labels = prepare_eval(cfg=cfg,
                            fold=fold,
                            metadata_path=os.path.join(original_cwd, cfg.data.metadata_path),
                            data_path=os.path.join(original_cwd, cfg.data.data_path),
                            checkpoint_path=checkpoint_path)

            accs.append(accuracy_score(labels, preds))
            f1_scores.append(f1_score(labels, preds, average='macro'))
            precisions.append(precision_score(labels, preds, average='macro'))
            recalls.append(recall_score(labels, preds, average='macro'))

        log.info(f"Accuracy mean: {np.mean(accs)} | Accuracy std: {np.std(accs)}")
        log.info(f"Recall mean: {np.mean(recalls)} | Recall std: {np.std(recalls)}")
        log.info(f"Precision mean: {np.mean(precisions)} | Precision std: {np.std(precisions)}")
        log.info(f"F1-Score mean: {np.mean(f1_scores)} | F1-Score std: {np.std(f1_scores)}")

    else:
        print("")


if __name__ == '__main__':
    main()