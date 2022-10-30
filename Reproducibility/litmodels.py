import torch
from torchmetrics.functional import accuracy

from pytorch_lightning import LightningModule


class LitClassifierModel(LightningModule):
    def __init__(self, model, optimizer, scheduler=None, num_classes=1000, lr=0.001, weight_decay=0, batch_size=32, warmup=None):
        super().__init__()
        # To satisfy a warning
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup = warmup
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
        top5_acc = accuracy(y_hat, y, top_k=5) * 100.
        # we use y_hat to display predictions during callback
        return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc, "test_top5_acc": top5_acc}

    # this is called at the end of all epochs
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_top5_acc = torch.stack([x["test_top5_acc"]
                                   for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)
        self.log("test_top5_acc", avg_top5_acc, on_epoch=True, prog_bar=True)

    # validation is the same as test
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def configure_optimizers(self):
        if self.scheduler is None:
            return eval(self.optimizer)
        else:
            optimizer = eval(self.optimizer)
            scheduler = eval(self.scheduler)
            return [optimizer], [scheduler]

    def configure_warmup(self):
        if self.warmup is None:
            return eval(self.warmup)
        else:
            warmup = eval(self.warmup)
            return [warmup]
