import torch.nn as nn
import torchvision.models as models

class ResnetV1(nn.Module):
    def __init__(self, num_classes):
        super(ResnetV1, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 4 parameters (typically layers close to the output)
        for param in list(self.resnet.parameters())[-4:]:
            param.requires_grad = True
        
        # Modify the last fully connected layer to match the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        return x
