import torch
import torch.nn as nn



## This is a convolutional block. This is 1 layer of a convolutional neural network
## A convolutional neural network is a neural network that precesses information with a sliding window inorder to extract extra information
## This block contains a Convolutional layer and a batch normalization layer (this layer helps prevent the network from memorizing and to actually generalize)
## the final blaock is an activation layer. this layer allows the model to learn more complicated fits to our data at a faster rate
class ConvBlockV1(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.batch_norm: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act: nn.Module = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.act(x)



class DefaultCNN(nn.Module):
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layer1: nn.Module = ConvBlockV1(1, 16, 3, 1, 1)
        self.maxpool: nn.Module = nn.MaxPool2d(2, 2)
        self.layer2: nn.Module = ConvBlockV1(16, 32, 3, 1,1 )
        self.layer3: nn.Module = ConvBlockV1(32, 64, 3, 1,1 )
        self.linear1: nn.LazyLinear = nn.LazyLinear(256)
        self.linear2: nn.LazyLinear = nn.LazyLinear(64)
        self.output: nn.LazyLinear = nn.LazyLinear(num_classes)
        self.relu = nn.ReLU()
        self.out_act: nn.Sigmoid = nn.Sigmoid()
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.layer1(x))
        x = self.maxpool(self.layer2(x))
        x = self.maxpool(self.layer3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.out_act(self.output(x))