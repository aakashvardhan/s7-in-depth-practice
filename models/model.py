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
    
if __name__ == '__main__':
    model = model1()
    model_summary(model, input_size=(1, 28, 28))
    test_model_sanity(model)
        