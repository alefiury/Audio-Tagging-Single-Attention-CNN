# Audio Tagging Single Attention CNN

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

- [Introduction](#Introduction)
- [How the Application Works](#How-the-Application-Works)
- [Contact](#Contact)

This repository contains an implementation in Pytorch of a Single Attention Convolutional Neural Network 
(**Single-Att-CNN**) model for audio tagging and sound event detection.

# Introduction

The model is trained and evaluated on the ESC-50 dataset, but can be easily modified to be
applied on others datasets or used as a base layout for your projects.

The model is comprised of a CNN component that is used as an encoder and a single-attention mechanism
that is based on how humans perceive and classify different kinds of sounds.


# How the Application Works

## Installation

Clone this repository and install all the dependencies.
The usage of a virtualenv or Conda environment is recommended.

```
git clone https://github.com/alefiury/Audio-Tagging-Single-Attention-CNN
cd Audio-Tagging-Single-Attention-CNN
pip install -r requirements.txt
```

## Dataset

First, is necessary to download the dataset that you will be using. The ESC-50 dataset can be
downloaded with the following bash command:

```
sudo bash download_dataset.sh
```

## Configuration

The application uses Hydra to control all the training configurations. For more
information on Hydra and it's documentation visit the Hydra [website](https://hydra.cc/).

The relevant information related with the training configuration can be found in the `default.yaml`
file, inside the `config` folder.

You can also pass or modify options through the command line, for instance: `python main.py train.epochs=100`.

## Logging

To save training statistics (loss, accuracy, gradients and etc) the Weights & Biases plataform is used.
Is necessary that you create an account before running the application.
Related information can be found on their [website](https://wandb.ai/site).

## Training and Evaluation

To train or evaluate your model you need to set the `command` option.

- To train: `python main.py command=train`
- To evaluate: `python main.py command=test`

# Author

- Alef Iury Siqueira Ferreira

# Contact

- e-mail: alef_iury_c.c@discente.ufg.br
