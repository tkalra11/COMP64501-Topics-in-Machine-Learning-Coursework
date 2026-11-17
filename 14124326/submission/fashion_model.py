import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name, and make sure Net() works with no arguments.
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model has 101,770 parameters, which is over the 100,000 limit.
    """
    def __init__(self, inputs = 1 , first_residual = 24 , first_transition =  44, second_transition = 64 , flatten = 32 , num_classes = 10):
        super(Net, self).__init__()

        # ResNET like structure with skip connections
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = inputs , out_channels = first_residual , kernel_size = 3 , padding = 1 , bias = False),
            nn.BatchNorm2d(first_residual),
            nn.ReLU(inplace = True)
        )

        # Residual Stage 1: (24, 28, 28) -> Residual -> MaxPool -> (24, 14, 14)
        self.res1 = nn.Sequential(
            nn.Conv2d(first_residual , first_residual , kernel_size = 3, padding = 1 , bias = False),
            nn.BatchNorm2d(first_residual),
            nn.ReLU(inplace = True),
            nn.Conv2d(first_residual , first_residual , kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24)
        )
        self.pool1 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2 , stride = 2)
        )

        # Transition + Residual Stage 2: (24, 14, 14) -> (44, 7, 7)
        self.trans2 = nn.Sequential(
            nn.Conv2d(first_residual , first_transition , kernel_size = 3 , padding = 1 , bias = False),
            nn.BatchNorm2d(first_transition),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(first_transition , first_transition , kernel_size = 3 , padding = 1 , bias = False),
            nn.BatchNorm2d(first_transition),
            nn.ReLU(inplace = True),
            nn.Conv2d(first_transition , first_transition , kernel_size = 3 , padding = 1 , bias = False),
            nn.BatchNorm2d(first_transition)
        )

        # Transition + Residual Stage 3: (44, 7, 7) -> (64, 3, 3)
        self.trans3 = nn.Sequential(
            nn.Conv2d(first_transition , second_transition , kernel_size = 3 , padding = 1 , bias = False),
            nn.BatchNorm2d(second_transition),
            nn.ReLU(inplace=True)
        )

        # Classifier: (64, 3, 3) -> 576 -> 32 -> 10
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_transition * 3 * 3, flatten),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(flatten , num_classes)
        )

    def forward(self, x):
        # Stage 1 with skip connection: x + res1(x)
        x = self.stem(x)
        x = self.pool1(x + self.res1(x))

        # Stage 2 with skip connection: x + res2(x)
        x = self.trans2(x)
        x = torch.relu(x + self.res2(x))

        # Stage 3 + Classifier
        x = self.pool3(self.trans3(x))
        logits = self.classifier(x)
        return logits