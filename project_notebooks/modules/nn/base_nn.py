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
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def forward(self) -> torch.Tensor:
        ...
        