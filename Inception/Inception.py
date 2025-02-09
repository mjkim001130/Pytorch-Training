import torch
import torch.nn as nn   
import torch.nn.functional as F


# -----------------------------------------------
# 1. Inception Block
# -----------------------------------------------
class Inception(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act = nn.ReLU):

        """
        c_in : number of input channels
        c_red : channel reduction 
        c_out : channel output
        """
        
        super().__init__()

        # 1x1 Convolution
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            act()
        )

        # 1x1 Convolution + 3x3 Convolution
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.ReLU(),
            act(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            act()
        )

        # 1x1 Convolution + 5x5 Convolution
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            act(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            act()
        )

        # 3x3 Maxpooling + 1x1 Convolution
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_out["maxpool"], kernel_size=1),
            act()
        )

    def forward(self, x):
        branch1 = self.conv_1x1(x)
        branch2 = self.conv_3x3(x)
        branch3 = self.conv_5x5(x)
        branch4 = self.maxpool(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)
    

# -----------------------------------------------
# 2. Auxiliary Classifier 
# -----------------------------------------------
class InceptionAux(nn.Module):

    def __init__(self, c_in, num_classes, dropout = 0.7, act = nn.ReLU):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 128, kernel_size=1),
            act()
        )
        # After avgpooling, and applying flatten, the shape is 128 x 4 x 4
        self.fc1 = nn.Linear(2048, 1024)
        self.act = act()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # auxiliar classifier
        
        # input : N x c_in x H x W
        x = self.avgpool(x)
        # N x 512 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# -----------------------------------------------
# 3. GoogleNet (Inception V1) 
# -----------------------------------------------
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux = True, dropout = 0.4, aux_dropout = 0.7):
        """
        GoogleNet (Inception V1) Architecture
        Args:
            num_classes : number of classes
            aux : auxiliary classifier
            dropout : dropout rate for the main classifier
            aux_dropout : dropout rate for the auxiliary classifier
        """
        super().__init__()

        self.aux = aux

        # Before Inception Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(64, 192,kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Blocks
        self.inception3a = Inception(
            c_in=192,
            c_red={"3x3": 96, "5x5": 16},
            c_out={"1x1": 64, "3x3": 128, "5x5": 32, "maxpool": 32},
            act=nn.ReLU
        )

        self.inception3b = Inception(
            c_in=256,
            c_red={"3x3": 128, "5x5": 32},
            c_out={"1x1": 128, "3x3": 192, "5x5": 96, "maxpool": 64},
            act=nn.ReLU
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(
            c_in=480,
            c_red={"3x3": 96, "5x5": 16},
            c_out={"1x1": 192, "3x3": 208, "5x5": 48, "maxpool": 64},
            act=nn.ReLU
        )
        if self.aux:
            self.aux1 = InceptionAux(512, num_classes, dropout=aux_dropout, act=nn.ReLU)
        else:
            self.aux1 = None

        self.inception4b = Inception(
            c_in=512,
            c_red={"3x3": 112, "5x5": 24},
            c_out={"1x1": 160, "3x3": 224, "5x5": 64, "maxpool": 64},
            act=nn.ReLU
        )

        self.inception4c = Inception(
            c_in=512,
            c_red={"3x3": 128, "5x5": 24},
            c_out={"1x1": 128, "3x3": 256, "5x5": 64, "maxpool": 64},
            act=nn.ReLU
        )

        self.inception4d = Inception(
            c_in=512,
            c_red={"3x3": 144, "5x5": 32},
            c_out={"1x1": 112, "3x3": 288, "5x5": 64, "maxpool": 64},
            act=nn.ReLU
        )

        if self.aux:
            self.aux2 = InceptionAux(528, num_classes, dropout=aux_dropout, act=nn.ReLU)
        else:
            self.aux2 = None

        self.inception4e = Inception(
            c_in=528,
            c_red={"3x3": 160, "5x5": 32},
            c_out={"1x1": 256, "3x3": 320, "5x5": 128, "maxpool": 128},
            act=nn.ReLU
        )

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(
            c_in=832,
            c_red={"3x3": 160, "5x5": 32},
            c_out={"1x1": 256, "3x3": 320, "5x5": 128, "maxpool": 128},
            act=nn.ReLU
        )

        self.inception5b = Inception(
            c_in=832,
            c_red={"3x3": 192, "5x5": 48},
            c_out={"1x1": 384, "3x3": 384, "5x5": 128, "maxpool": 128},
            act=nn.ReLU
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        # Before Inception Blocks
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        # Inception Blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        aux1 = None
        if self.aux and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = None
        if self.aux and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux1, aux2

        
                                                                                                                                                                                                                                                                                                                                                    