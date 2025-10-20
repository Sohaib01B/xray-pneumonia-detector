import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(PneumoniaClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(device, num_classes=2):
    model = PneumoniaClassifier(num_classes=num_classes)
    model = model.to(device)
    return model