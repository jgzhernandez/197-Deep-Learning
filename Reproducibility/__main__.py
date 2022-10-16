import torch
from torchvision import models

import os
from argparse import ArgumentParser

import wandb

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from litmodels import LitClassifierModel
from litdataloader import ImageNetDataModule


class WandbCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        global wandb_logger
        # process first 10 images of the first batch
        if batch_idx == 0:
            n = 10
            x, y = batch
            outputs = outputs["y_hat"]
            outputs = torch.argmax(outputs, dim=1)
            # log image, ground truth and prediction on wandb table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(
                zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(
                key=f'{args.surname[0].capitalize()} on ImageNet Predictions',
                columns=columns,
                data=data)


def get_args():
    parser = ArgumentParser(
        description="PyTorch Lightning Classifier Example on ImageNet1k")
    parser.add_argument("--surname", type=str,
                        default="resnet18", help="surname")

    parser.add_argument("--max-epochs", type=int, default=5, help="num epochs")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")

    parser.add_argument("--path", type=str, default="./")

    parser.add_argument("--num-classes", type=int,
                        default=1000, help="num classes")

    parser.add_argument("--devices", default=1)
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--num-workers", type=int,
                        default=48, help="num workers")

    parser.add_argument("--no-wandb", default=False, action='store_true')
    args = parser.parse_args("")
    return args


def resnet18(num_classes):
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)
    return model


def atienza(num_classes):
    return resnet18(num_classes)


if __name__ == "__main__":
    args = get_args()

    model_selector = {
        "resnet18": resnet18,
        "atienza": atienza,
    }

    model = LitClassifierModel(model=model_selector[args.surname],
                               num_classes=args.num_classes,
                               lr=args.lr, batch_size=args.batch_size)
    datamodule = ImageNetDataModule(
        path=args.path, batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup()

    # printing the model is useful for debugging
    print(model)

    # wandb is a great way to debug and visualize this model
    wandb_logger = WandbLogger(project=f"pl-{args.surname}")

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename=f"{args.surname}-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger if not args.no_wandb else None,
                      callbacks=[model_checkpoint, WandbCallback() if not args.no_wandb else None])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()
