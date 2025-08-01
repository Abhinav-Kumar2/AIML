import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module): #inherting to make a custom network architecture from the base nn.module provided by pytorch
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__() #initializing the base pytorch class nn.Module
        
        self.features = nn.Sequential( #to extract features from the input image
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),                

            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),                

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)               
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True),

            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
