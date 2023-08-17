"""
Implementation of YOLOv3 architecture
"""

from typing import Any, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning as L

import config as config_
from utils.common import one_cycle_lr
from utils.data import PascalDataModule
from utils.loss import YoloLoss
from utils.utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
)


""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(L.LightningModule):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(L.LightningModule):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(L.LightningModule):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(L.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=80,
        epochs=40,
        loss_fn=YoloLoss,
        datamodule=PascalDataModule(),
        learning_rate=None,
        maxlr=None,
        scheduler_steps=None,
        device_count=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.epochs = epochs
        self.loss_fn = loss_fn()
        self.layers = self._create_conv_layers()
        self.scaled_anchors = torch.tensor(config_.ANCHORS) * torch.tensor(
            config_.S
        ).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(self.device)
        self.datamodule = datamodule
        self.learning_rate = learning_rate
        self.maxlr = maxlr
        self.scheduler_steps = scheduler_steps
        self.device_count = device_count

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3

        return layers

    def configure_optimizers(self) -> Dict:
        # effective_lr = self.learning_rate * self.device_count
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=config_.WEIGHT_DECAY
        )
        scheduler = one_cycle_lr(
            optimizer=optimizer,
            maxlr=self.maxlr,
            steps=self.scheduler_steps,
            epochs=self.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
        
    def _common_step(self, batch, batch_idx):
        self.scaled_anchors = self.scaled_anchors.to(self.device)
        x, y = batch
        y0, y1, y2 = y[0], y[1], y[2]
        out = self(x)
        loss = (
            self.loss_fn(out[0], y0, self.scaled_anchors[0])
            + self.loss_fn(out[1], y1, self.scaled_anchors[1])
            + self.loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log(name="train_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log(name="val_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        class_acc, noobj_acc, obj_acc = check_class_accuracy(
            model=self,
            loader=self.datamodule.test_dataloader(),
            threshold=config_.CONF_THRESHOLD,
        )

        self.log_dict(
            {
                "class_acc": class_acc,
                "noobj_acc": noobj_acc,
                "obj_acc": obj_acc,
            },
            prog_bar=True,
        )


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (
        2,
        3,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        num_classes + 5,
    )
    assert model(x)[1].shape == (
        2,
        3,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        num_classes + 5,
    )
    assert model(x)[2].shape == (
        2,
        3,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
        num_classes + 5,
    )
    print("Success!")
