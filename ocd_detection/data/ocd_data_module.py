import os
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

class OCDDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None, data_dir='./dataset'):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.data_dir = data_dir
        self.dims = (3, 244, 244)
        self.output_dims = (1,)
        self.mapping = list(range(2))

        self.transforms = {
            'train': T.Compose([
                T.Resize(size=self.dims[1]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            'val': T.Compose([
                T.Resize(size=self.dims[1]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            'test': T.Compose([
                T.Resize(size=self.dims[1]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])            
        }

        self.train_path = os.path.join(data_dir, 'train')
        self.val_path = os.path.join(data_dir, 'valid')
        self.test_path = os.path.join(data_dir, 'test')

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser 

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate the model."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self):
        """
        No downloading needed so we can skip
        """
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageFolder(self.train_path, transform=self.transforms['train'])
            self.val_dataset = datasets.ImageFolder(self.val_path, transform=self.transforms['val'])

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.ImageFolder(self.test_path, transform=self.transforms['test'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )