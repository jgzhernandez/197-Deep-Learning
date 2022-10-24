import torch
from torchvision import models

from torchvision.models import SqueezeNet1_1_Weights

import os
from argparse import ArgumentParser

import numpy as np

import wandb

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from litmodels import LitClassifierModel
from litdataloader import ImageNetDataModule
from classnames import CLASS_NAMES_LIST


class WandbCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        global wandb_logger
        # process (batch_size // 10) random images of of every 5th batch (ensure 60 images for 128)
        if batch_idx % 5 == 0 and batch_idx <= (5*(128//args.batch_size))*5:
            n = args.batch_size // 10
            x, y = batch
            outputs = outputs["y_hat"]
            outputs = torch.argmax(outputs, dim=1)

            # generate n random integers from 0 to len(x)
            random_image = np.random.randint(0, len(x), n)

            classes_to_idx = pl_module.hparams.classes_to_idx
            # log image, ground truth and prediction on wandb table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), classes_to_idx[int(y_i)], classes_to_idx[int(y_pred)]] for x_i, y_i, y_pred in list(
                zip([x[i] for i in random_image], [y[i] for i in random_image], [outputs[i] for i in random_image]))]
            wandb_logger.log_table(
                key=f'{args.surname.capitalize()} on ImageNet Predictions (Batch {batch_idx})',
                columns=columns,
                data=data)


def get_args():
    parser = ArgumentParser(
        description="PyTorch Lightning Classifier Example on ImageNet1k")
    parser.add_argument("--surname", type=str,
                        default="resnet18", help="surname")

    ARG_DEFAULTS = {
        "--max-epochs": 100,
        "--batch-size": 32,
        "--lr": 0.001,
        "--weight-decay": 0,
        "--path": "./",
        "--num-classes": 1000,
        "--devices": [0],
        "--accelerator": "gpu",
        "--num-workers": 48,
        "--optimizer": "Adam",
    }

    parser.add_argument("--max-epochs", type=int, help="num epochs")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight-decay", type=float, help="weight decay")

    parser.add_argument("--path", type=str)

    parser.add_argument("--num-classes", type=int, help="num classes")

    parser.add_argument("--devices", type=int, nargs=1)
    parser.add_argument("--accelerator")
    parser.add_argument("--num-workers", type=int, help="num workers")

    parser.add_argument("--optimizer", type=str)

    args = parser.parse_args()

    print(f"surname: {args.surname}")
    for key, default_value in ARG_DEFAULTS.items():
        arg_name = "_".join(key.split("-")[2:])
        arg_text = " ".join(key.split("-")[2:])
        if args.__getattribute__(arg_name) is None:
            args.__setattr__(arg_name, default_value)
        elif args.__getattribute__(arg_name) != default_value:
            print(f"{arg_text}: {args.__getattribute__(arg_name)}")

    return args


def resnet18(num_classes):
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)
    return model


def atienza(num_classes):
    return resnet18(num_classes)


def ancheta(num_classes):
    # SqueezeNet 1.1
    # python Reproducibility --surname ancheta --max-epochs 100 --lr 0.01 --weight-decay 0.0002 --batch-size 128 --optimizer AdamW
    return models.squeezenet1_1()


if __name__ == "__main__":
    args = get_args()

    # Add your model here
    model_selector = {
        "resnet18": resnet18,
        "atienza": atienza,
        "ancheta": ancheta,
    }

    # Add the transforms in your recipe, litdataloader has its own
    # but it's recommended to use the transforms in your recipe
    transform_selector = {
        "ancheta": SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms(),
    }

    # Sometimes accuracy barely changes so you should choose
    # a different optimizer from default (Adam)
    optimizer_selector = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }

    classes_to_idx = CLASS_NAMES_LIST

    model = LitClassifierModel(model=model_selector[args.surname](args.num_classes),
                               optimizer=optimizer_selector[args.optimizer],
                               num_classes=args.num_classes,
                               lr=args.lr, weight_decay=args.weight_decay,
                               batch_size=args.batch_size)
    datamodule = ImageNetDataModule(
        path=args.path, batch_size=args.batch_size, num_workers=args.num_workers,
        class_dict=classes_to_idx,
        transform=transform_selector.get(args.surname))
    datamodule.setup()

    # printing the model is useful for debugging
    print(model)

    # wandb is a great way to debug and visualize this model
    wandb_logger = WandbLogger(project=f"reproducibility-pl-{args.surname}")

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.path, "checkpoints"),
        filename=f"reproducibility-{args.surname}-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger,
                      callbacks=[model_checkpoint, WandbCallback()])
    model.hparams.classes_to_idx = classes_to_idx
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()
