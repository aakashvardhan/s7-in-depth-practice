import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model_utils import model_summary
from model_utils import test_model_sanity


class model1(nn.Module):
    def __init__(self,n_channels=20):
        super().__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_channels // 2, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU()
        ) # output_size = 26, RF = 3
        
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU()
        ) # output_size = 24, RF = 5
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        ) # output_size = 22, RF = 7
        
        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU()
        ) # output_size = 11, RF = 8
        
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU()
        ) # output_size = 9, RF = 12
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        ) # output_size = 7, RF = 16
        
        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU()
        ) # output_size = 7, RF = 16
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels // 2, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
        ) # output_size = 1, RF = 28
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),dropout_value=0, **kwargs):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,bias=False, **kwargs),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,1), **kwargs):
        super().__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs),
        )
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv1d(x)
        return self.pool(x)
    
class model2(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        
        # Convolution Block 1
        self.conv1 = ConvBlock(1, n_channels // 2) # output_size = 26, RF = 3
        self.conv2 = ConvBlock(n_channels // 2, n_channels) # output_size = 24, RF = 5
        self.conv3 = ConvBlock(n_channels, n_channels) # output_size = 22, RF = 7
        
        # Transition Block 1
        self.transition1 = TransitionBlock(n_channels, n_channels // 2) # output_size = 11, RF = 8
        
        # Convolution Block 2
        self.conv4 = ConvBlock(n_channels // 2, n_channels // 2, dropout_value=0.1) # output_size = 9, RF = 12
        self.conv5 = ConvBlock(n_channels // 2, n_channels, dropout_value=0.1) # output_size = 7, RF = 16
        self.conv6 = ConvBlock(n_channels, n_channels, padding=1) # output_size = 7, RF = 16
        
        # Output Block
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) # output_size = 1, RF = 28
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF = 28
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transition1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
            
class model3(nn.Module):
    def __init__(self, n_channels=12):
        super().__init__()
        
        # Convolution Block 1
        self.conv1 = ConvBlock(1, n_channels // 2) # output_size = 26, RF = 3
        self.conv2 = ConvBlock(n_channels // 2, n_channels // 3, dropout_value=0.1) # output_size = 24, RF = 5
        self.conv3 = ConvBlock(n_channels // 3, n_channels, dropout_value=0.1) # output_size = 22, RF = 7
        
        # Transition Block 1
        self.transition1 = TransitionBlock(n_channels, n_channels // 2) # output_size = 11, RF = 8
        
        # Convolution Block 2
        self.conv4 = ConvBlock(n_channels // 2, n_channels // 2, dropout_value=0.1) # output_size = 9, RF = 12
        self.conv5 = ConvBlock(n_channels // 2, n_channels // 3, dropout_value=0.1) # output_size = 7, RF = 16
        self.conv6 = ConvBlock(n_channels // 3, n_channels, dropout_value=0.1) # output_size = 5, RF = 20
        
        # Output Block
        self.adapool = nn.AdaptiveAvgPool2d((1,1)) # output_size = 1, RF = 28
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF = 28
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transition1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.adapool(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
            
    
if __name__ == '__main__':
    model = model3()
    model_summary(model, input_size=(1, 28, 28))
    test_model_sanity(model)
        