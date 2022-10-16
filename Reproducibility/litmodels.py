import torch
from torchvision import models
from torchmetrics.functional import accuracy

from pytorch_lightning import LightningModule


class LitClassifierModel(LightningModule):
    def __init__(self, model=models.resnet18(num_classes=10), num_classes=10, lr=0.001, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7,
                                           stride=2, padding=3, bias=False)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    # this is called during fit()
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    # calls to self.log() are recorded in wandb
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)

    # this is called at the end of an epoch
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y) * 100.
        # we use y_hat to display predictions during callback
        return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc}

    # this is called at the end of all epochs
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)

    # validation is the same as test
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    # we use Adam optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
