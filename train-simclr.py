import sys
from augment import Augmentation
from cowDataset import CustomDataset
from sampler import CustomSampler
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    random_split
)
from torchvision import datasets

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from simCLR import SimCLR
import datetime

import os
from split_data import split

NUM_WORKERS = os.cpu_count()
EPOCHS = 100


def train_sequence(model, trainer, train_dataloader, validation_dataloader):
    pl.seed_everything(42)
    trainer.fit(model, train_dataloader, validation_dataloader)

    return model


def load_dataset(train, dir, label_map):
    ds = CustomDataset(dir, train, label_map, Augmentation())

    train_size = int(len(ds) * 0.85)
    val_size = len(ds) - train_size

    training_set, validation_set = random_split(ds, [train_size, val_size])

    return training_set, validation_set


def create_dataloader(dataset, shuffle):

    sampler = CustomSampler(data=dataset, batch_size=128,
                            shuffle=shuffle, num_replicas=1, rank=0)

    dataloader = DataLoader(dataset,
                            batch_sampler=sampler, persistent_workers=True, num_workers=5)

    return dataloader


def generate_label_map(dir):
    label_map = {}
    index = 0

    for date in sorted(os.listdir(dir)):
        date_path = os.path.join(dir, date)

        for camera in sorted(os.listdir(date_path)):
            camera_path = os.path.join(date_path, camera)

            for idx in sorted(os.listdir(camera_path)):
                label_map[(date, camera, idx)] = index
                index += 1
    return label_map


def generate_data_split(dir, save_dir):
    train = np.array([])
    test = np.array([])
    dates_available = os.listdir(dir)
    for date in dates_available:
        curr_train, curr_test = split(f'{dir}/{date}')
        train = np.concatenate((train, curr_train))
        test = np.concatenate((test, curr_test))

    with open(f'{save_dir}/train_data_paths.txt', 'w') as f:
        np.savetxt(f, train, fmt="%s")
        print(f"Successfully saved {len(train)} items to Train")
    with open(f'{save_dir}/test_data_paths.txt', 'w') as f:
        np.savetxt(f, test, fmt="%s")
        print(f"Successfully saved {len(test)} items to Test")
    return train, test


if __name__ == "__main__":

    data_dir = f'{sys.argv[1]}/images'
    save_dir = f'{sys.argv[1]}/models'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model = SimCLR(epochs=EPOCHS)
    model.train()

    logger = TensorBoardLogger(
        save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=1,
        monitor="Total_Validation_NTXent_loss",
        filename="best-model-checkpoint",
        every_n_epochs=1)

    earlystop_callback = EarlyStopping(
        monitor=f"Total_Validation_NTXent_loss",
        patience=10,
        verbose=True,
        mode="min")

    trainer = pl.Trainer(max_epochs=EPOCHS,
                         log_every_n_steps=30,
                         default_root_dir=save_dir,
                         accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint_callback, earlystop_callback])

    label_map = generate_label_map(data_dir)
    train, test = generate_data_split(data_dir, save_dir)

    train_dataset, validation_dataset = load_dataset(
        train, data_dir, label_map)

    train_dataloader = create_dataloader(train_dataset, True)
    validation_dataloader = create_dataloader(validation_dataset, False)

    model = train_sequence(
        model, trainer, train_dataloader, validation_dataloader)
