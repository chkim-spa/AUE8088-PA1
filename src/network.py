# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy
from src.metric import MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, num_classes: int = 200, dropout: float = 0.5) -> None:
        super().__init__()
        # [TODO] Modify feature extractor part in AlexNet

        self.features = nn.Sequential(
            # kernel_size 11을 3으로 축소
            # (11, 4, 2): 64->15
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),    # (3, 2, 1): 64 -> 33
            nn.BatchNorm2d(64),                                      # add BN       
            nn.GELU(),                                               # GELU는 'inplace=True'인자가 없음
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),   # (3, 2, 1): 33 -> 30
            nn.GELU(),                                               # ReLU->GELU
            nn.BatchNorm2d(64),                                      # add BN
            nn.MaxPool2d(kernel_size=3, stride=2),                   # (3, 2, 1): 30 -> 15
 
            # kernel_size 5을 3으로 축소 
            # (5, 1, 2): 7->7 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (3, 2, 1): 15 -> 7
            nn.GELU(), 
            nn.Conv2d(128, 192, kernel_size=3,padding=1),            # (3, 2, 1): 7 -> 7  
            nn.GELU(),                                               # ReLU->GELU
            nn.BatchNorm2d(192),                                     # add BN
            nn.MaxPool2d(kernel_size=3, stride=2), 
 
            nn.Conv2d(192, 384, kernel_size=3, padding=1), 
            nn.GELU(),                                               # ReLU->GELU
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.GELU(),                                               # ReLU->GELU
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.GELU(),                                               # ReLU->GELU
            nn.BatchNorm2d(256),                                     # add BN
            # nn.MaxPool2d(kernel_size=3, stride=2),                   # (3, 2, 0): 7 -> 2
        )
        self.seblock = SEBlock(channel=256, reduction=16)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.seblock(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.F1Score  = MyF1Score()

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler':scheduler, 'monitor':'loss/val'}}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        F1Score  = self.F1Score(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'F1Score/train' : F1Score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        F1Score  = self.F1Score(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'F1Score/val' : F1Score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
