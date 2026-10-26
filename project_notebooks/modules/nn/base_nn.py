import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor of shape [num_classes] indicating weights (inverse frequency)
        gamma: Focusing parameter. Higher = less attention to easy samples (Default: 2)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # Apply class weights to the focal loss
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



## This is a convolutional block. This is 1 layer of a convolutional neural network
## A convolutional neural network is a neural network that precesses information with a sliding window inorder to extract extra information
## This block contains a Convolutional layer and a batch normalization layer (this layer helps prevent the network from memorizing and to actually generalize)
## the final blaock is an activation layer. this layer allows the model to learn more complicated fits to our data at a faster rate
class ConvBlockV1(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.batch_norm: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.act: nn.Module = nn.LeakyReLU(inplace=True)
        self.max_pool: nn.MaxPool2d = nn.MaxPool2d(2,2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.max_pool(self.act(x))
    
    
class DepthwiseSeparableBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block: nn.Sequential = nn.Sequential(
            # 1. DEPTHWISE STEP: Spatial filtering (3x3)
            # groups=in_channels forces each channel to be filtered independently
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            
            # 2. POINTWISE STEP: Channel mixing (The 1x1 Point After)
            # This blends the filtered channels together into the new channel size
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DefaultCNN(nn.Module):
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.convs: nn.Sequential = nn.Sequential(
            DepthwiseSeparableBlock(3, 32),
            DepthwiseSeparableBlock(32, 64),
            DepthwiseSeparableBlock(64, 128),
        )
        
        self.classifier: nn.Sequential = nn.Sequential(
            # ponytail: GAP instead of Flatten — kills the 4.85M-param lookup-table head
            # (18944->256 Linear) that was memorizing 9k train segments; 128->256 is ~33k params.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.LazyLinear(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(num_classes),
            #nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.classifier(x)
        return x
    
    
class BranchedCoughCNN(nn.Module):
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        
        self.spec_branch: nn.Sequential = nn.Sequential(
            DepthwiseSeparableBlock(3, 32),
            DepthwiseSeparableBlock(32, 64)
        )
        
        self.mfcc_branch: nn.Sequential = nn.Sequential(
            DepthwiseSeparableBlock(3, 32),
            DepthwiseSeparableBlock(32, 64)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier: nn.Sequential = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            #nn.Sigmoid()
        )
        
    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        
        mel_out = self.spec_branch(mel)       # Shape: [B, 64, H_spec, W_spec]
        mfcc_out = self.mfcc_branch(mfcc)     # Shape: [B, 64, H_mfcc, W_mfcc]
        
        # Apply global pooling so size mismatches in H or W across batches don't break things
        mel_out = self.pool(mel_out)         # Shape: [B, 64, 1, 1]
        mfcc_out = self.pool(mfcc_out)       # Shape: [B, 64, 1, 1]
        
        # Concatenate along the channel dimension (dim=1)
        combined = torch.cat([mel_out, mfcc_out], dim=1)  # Shape: [B, 128, 1, 1]
        
        # Flatten from 4D to 2D for the linear classifier: [B, 128]
        combined = torch.flatten(combined, start_dim=1)
        
        return self.classifier(combined)

 
class ResNetModel(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)