import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from convkan.kanlinear import KANLinear
from convkan import ConvKAN, LayerNorm2D

def _pair(x):
    if isinstance(x, (int, float)):
        return x, x
    return x

class Imagenet200DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int = 32, num_workers: int = 1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform
        )
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform
        )
        self.test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "test"), transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ConvKANLitModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            ConvKAN(3, 32, padding=1, kernel_size=3, stride=1),
            LayerNorm2D(32),
            ConvKAN(32, 64, padding=1, kernel_size=3, stride=2),
            LayerNorm2D(64),
            ConvKAN(64, 200, padding=1, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('test_acc', acc)


if __name__ == '__main__':
    DATA_DIR = "/data/pgreshnikov/tiny-imagenet-200"
    BATCH_SIZE = 32
    EPOCHS = 100

    dm = Imagenet200DataModule(DATA_DIR, batch_size=BATCH_SIZE)
    model = ConvKANLitModel(lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10,
    )
    print(f"Device: {trainer.strategy.root_device}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters())}")
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
