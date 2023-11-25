from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models
import torch
import torchmetrics
from torch import nn, optim
import numpy as np
import lightning.pytorch as pl
import cv2


# Архитектура модели
class LitModel(pl.LightningModule):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.convnext_tiny(weights="IMAGENET1K_V1")
        self.model.classifier[2] = nn.Linear(768, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        self.log('test_acc', acc)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

# Загружаем веса в модель
inference_model = LitModel.load_from_checkpoint("best_model.ckpt")
inference_model.eval()


def classification_trash(frame):
    ''' Функция принимает на вход frame OpenCV, делает классификацию с
        помощью предобученной модели и выдаёт следующие параметры:
         :prob_dict - словарь распределение вероятности классов (dict)
         :label - обнаруженный вид отхода (str)
    '''
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                                   ])
    
    image = transform(frame).unsqueeze(0)

    with torch.no_grad():
        pred = inference_model(image)

    # prob_dict - словарь вероятностей по видам отхода
    prob = F.softmax(pred, dim=1)
    prob = [round(float(x), 3) for x in list(prob[0])]
    prob_dict = {key: p for key, p in zip(['bricks', 'concrete', 'earth', 'tree'], list(prob))}

    # label - итог классификации, какой класс отхода
    class_to_index = {0: 'bricks',
                    1: 'concrete',
                    2: 'earth',
                    3: 'tree'
                    }
    label = class_to_index[torch.argmax(pred, dim=1).item()]

    return prob_dict, label
