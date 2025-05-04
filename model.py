import torch
import torch.nn as nn

class PatchCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PatchCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 96, kernel_size=9, stride=3, padding=0)
        self.bn1 = nn.BatchNorm2d(96)  # Using BatchNorm instead of LRN
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)  # Using BatchNorm instead of LRN
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional block
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((15, 15))
        
        # Calculate the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 400, 400)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.relu3(self.conv3(x))
            x = self.relu4(self.conv4(x))
            x = self.pool5(self.relu5(self.conv5(x)))
            x = self.adaptive_pool(x)
            fc_input_size = x.view(1, -1).size(1)
            print(f"Feature map size after convolutions: {fc_input_size}")
        
        # Create classifier with dynamically calculated input size
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool5(self.relu5(self.conv5(x)))

        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
