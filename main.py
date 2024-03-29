import os
import glob
import logging
import argparse

import wandb
import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from utils.utils import formatter_single
from utils.prepare_train_eval import prepare_train, prepare_eval

device = ('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login()

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("config", "default.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='If True, train model'
    )
    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='If True, test model'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if args.train:
        wandb_config = {
            **cfg.train,
            **cfg.model,
            **cfg.data,
        }

        for fold in cfg.train.folds:
            log.info(f"Training fold: {fold} -> {len(cfg.train.folds)}... ")
            # Start a new run on wandb
            run = wandb.init(
                project=os.path.basename(cfg.data.output_dir),
                config=wandb_config,
                reinit=True,
                mode=None if cfg.wandb_logging else "disabled"
            )

            # Change run name to facilitate statistics traceability
            wandb.run.name = 'fold '+str(fold)
            wandb.run.save()

            prepare_train(
                cfg=cfg,
                fold=fold,
                metadata_path=cfg.data.metadata_path,
                data_path=cfg.data.data_path,
                output_path=os.path.join(cfg.data.output_dir, 'fold_'+str(fold))
            )

            # Finish run
            run.finish()


    elif args.test:
        accs = []
        f1_scores = []
        precisions = []
        recalls = []
        for fold in cfg.train.folds:
            log.info(f"Testing fold: {fold} -> {len(cfg.train.folds)}... ")

            # Get saved weights paths
            checkpoint_path = glob.glob(os.path.join(cfg.data.output_dir, 'fold_'+str(fold), '*.pth'))[0]

            preds, labels = prepare_eval(
                cfg=cfg,
                fold=fold,
                metadata_path=cfg.data.metadata_path,
                data_path=cfg.data.data_path,
                checkpoint_path=checkpoint_path
            )

            accs.append(accuracy_score(labels, preds))
            f1_scores.append(f1_score(labels, preds, average='macro'))
            precisions.append(precision_score(labels, preds, average='macro'))
            recalls.append(recall_score(labels, preds, average='macro'))

        log.info(f"Accuracy mean: {np.mean(accs)} | Accuracy std: {np.std(accs)}")
        log.info(f"Recall mean: {np.mean(recalls)} | Recall std: {np.std(recalls)}")
        log.info(f"Precision mean: {np.mean(precisions)} | Precision std: {np.std(precisions)}")
        log.info(f"F1-Score mean: {np.mean(f1_scores)} | F1-Score std: {np.std(f1_scores)}")

    else:
        log.info("Wrong Command... Exiting... ")


if __name__ == '__main__':
    main()
