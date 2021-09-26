import os
import logging

import torch
import wandb
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from omegaconf import DictConfig

from utils.utils import Mixup, do_mixup
from utils.losses import BCELossModified
from utils.scores import model_accuracy, model_f1_score
from model import Cnn_Single_Att, Wavegram_Logmel_Cnn_Single_Att

device = ('cuda' if torch.cuda.is_available() else 'cpu')
# Logger
log = logging.getLogger(__name__)

def train_model(train_dataloader: torch.utils.data.DataLoader,
                    val_dataloader: torch.utils.data.DataLoader,
                    output_dir: str,
                    cfg : DictConfig) -> None:

    # Create directory to save weights
    os.makedirs(output_dir, exist_ok=True)

    if cfg.data.use_mixup:
        log.info('Using Mix up')
    if cfg.data.use_specaug:
        log.info('Using SpecAugment')
    if cfg.model.imagenet_pretrained:
        log.info('ImageNet Pre-Trained')

    if cfg.train.use_wavgram_logmel:
        log.info('Using Wavgram Logmel Model')
        model = Wavegram_Logmel_Cnn_Single_Att(**cfg.model)
    else:
        log.info('Using Single Attention CNN Model')
        model = Cnn_Single_Att(**cfg.model)

    model.to(device)

    # Using same loss function as PANNs
    criterion = BCELossModified()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=cfg.train.lr,
                                amsgrad=True)

    # Save gradients of the weights
    wandb.watch(model, criterion, log='all', log_freq=10)

    loss_min = np.Inf

    # Initialize early stopping
    p = 0

    # Initialize scheduler
    num_train_steps = int(len(train_dataloader) * cfg.train.epochs)
    num_warmup_steps = int(0.1 * cfg.train.epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    mix_up = Mixup(mixup_alpha=cfg.data.mixup_alpha)

    model.train()

    for e in tqdm(range(cfg.train.epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in enumerate(train_dataloader):
            train_audio, train_label = sample['image'].to(device), sample['hot_target'].to(device)

            optimizer.zero_grad()

            # Mix up
            if cfg.data.use_mixup:
                mixup_lambda = mix_up.get_lambda(batch_size=cfg.train.batch_size)
                out = model(train_audio,
                                use_specaug=cfg.data.use_specaug,
                                mixup_lambda=mixup_lambda)
                train_label = do_mixup(train_label, mixup_lambda)

            else:
                out = model(train_audio, use_specaug=cfg.data.use_specaug)

            train_loss = criterion(out['clipwise_output'], train_label)
            train_loss.backward()
            optimizer.step()

            if scheduler and cfg.data.step_scheduler:
                scheduler.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(train_label, out)
            train_f1_score += model_f1_score(train_label, out)

            train_batch_count += 1

            if (train_batch_count % 2) == 0:
                wandb.log({"train_loss": train_loss})

        else:

            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            with torch.no_grad():
                model.eval()

                for val_batch_count, val_sample in enumerate(val_dataloader):
                    val_audio, val_label = val_sample['image'].to(device), val_sample['hot_target'].to(device)

                    out = model(val_audio)

                    loss = criterion(out['clipwise_output'], val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            # Log results on wandb
            wandb.log({"train_acc": (train_accuracy/len(train_dataloader))*100,
                        "val_acc": (val_accuracy/len(val_dataloader))*100,
                        "train_f1": train_f1_score/len(train_dataloader)*100,
                        "val_f1": val_f1_score/len(val_dataloader)*100,
                        "epoch": e})

            log.info('Train Accuracy: {:.3f} | Train F1-Score: {:.3f} | Train Loss: {:.6f} | Val Accuracy: {:.3f} | Val F1-Score: {:.3f} | Val loss: {:.6f}'.format(
                        (train_accuracy/len(train_dataloader))*100,
                        (train_f1_score/len(train_dataloader))*100,
                        running_loss/len(train_dataloader),
                        (val_accuracy/len(val_dataloader))*100,
                        (val_f1_score/len(val_dataloader))*100,
                        val_loss/len(val_dataloader)))

            log.info(f"LR: {scheduler.get_last_lr()[0]}")

            if val_loss/len(val_dataloader) < loss_min:
                log.info("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save({'epoch': cfg.train.epochs,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': criterion
                            }, os.path.join(output_dir, f'epochs_{cfg.train.epochs}.pth'))

            else:
                p += 1

                if p == cfg.train.patience:
                    log.info("Early Stopping... ")
                    break

            model.train()