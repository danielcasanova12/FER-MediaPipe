from torchvision import models
import torch.nn as nn

class CustomResNet50:
    def __init__(self, num_classes):
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def get_model(self):
        return self.model
