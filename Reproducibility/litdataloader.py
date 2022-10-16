import torch

from torchvision import transforms, datasets
from pytorch_lightning import LightningDataModule


class ImageNetDataModule(LightningDataModule):
    def __init__(self, path, batch_size=128, num_workers=0, class_dict={},
                 **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_dict = class_dict

    def prepare_data(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
        ])

        self.train_dataset = datasets.ImageNet(
            "/data/imagenet/", split='train', transform=self.transform)

        # self.val_dataset = datasets.ImageNet(
        #     "/data/imagenet/", split='train', transform=self.transform)

        self.test_dataset = datasets.ImageNet(
            "/data/imagenet/", split='val', transform=self.transform)

    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #     )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
