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


class SpatialAttention(nn.Module):

  def __init__(self, kernel_size: int = 7) -> None:
    super().__init__()
    assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
    
    padding: int = 3 if kernel_size == 7 else 1
    
    self.conv: nn.Conv2d = nn.Conv2d(
        2, 1, kernel_size=kernel_size, padding=padding, bias=False
    )
    
    self.sigmoid: nn.Sigmoid = nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    avg_out: torch.Tensor = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    pooled: torch.Tensor = torch.cat([avg_out, max_out], dim=1)
    attention_map: torch.Tensor = self.sigmoid(self.conv(pooled))
    return x * attention_map

    
class DepthwiseSeparableBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 4, att_kernel: int = 7) -> None:
        super().__init__()
        # Inverted bottleneck (MobileNetV2 style): EXPAND channels with a 1x1,
        # filter in the widened space, PROJECT back down with a linear 1x1.
        exp_channels: int = in_channels * expansion
        self.block: nn.Sequential = nn.Sequential(
            # 0. EXPANSION STEP: widen channels before the depthwise filter
            nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU6(inplace=True),

            # 1. DEPTHWISE STEP: Spatial filtering (3x3) in the widened space
            # groups=exp_channels forces each channel to be filtered independently
            nn.Conv2d(exp_channels, exp_channels, kernel_size=3, padding=1, stride=stride, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU6(inplace=True),

            # 2. POINTWISE STEP: Channel mixing (The 1x1 Point After)
            # Linear projection — no activation, so the compression is unclipped
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SpatialAttention(att_kernel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _branch_and_skips(specs: list[tuple[int, int, int | None]], expansion: int = 4) -> tuple[nn.ModuleList, nn.ModuleList]:
    """A stage list plus a 1x1-conv skip projection per stage.

    The skip path lives at the CNN level, not inside the block: each stage's
    input is projected (1x1 conv + BN, only to match stride/channel shape) and
    added back after the stage, so the skip goes AROUND the depthwise path.
    """
    blocks = nn.ModuleList(
        DepthwiseSeparableBlock(
            i, o, stride=2 if not s else s, 
            att_kernel=7 if o <= 64 else 3, 
            expansion=expansion
        ) for i, o, s in specs
    )
    skips = nn.ModuleList(
        nn.Sequential(
            nn.Conv2d(i, o, kernel_size=1, stride=s if s else 2, bias=False),
            nn.BatchNorm2d(o)
        )
        for i, o, s in specs
    )
    return blocks, skips


def _res_forward(blocks: nn.ModuleList, skips: nn.ModuleList, x: torch.Tensor,  drop: nn.Module | None = None) -> torch.Tensor:
    for block, skip in zip(blocks, skips):
        x = F.relu6(block(x) + skip(x))
        if drop is not None:
            x = drop(x)
    return x


class DefaultCNN(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.convs, self.skips = _branch_and_skips([(3, 32, 2), (32, 64, 2), (64, 128, 1)], 5)
        self.drop2d: nn.Dropout2d = nn.Dropout2d(0.1)
        self.classifier: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU6(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _res_forward(self.convs, self.skips, x, self.drop2d)
        x = self.classifier(x)
        return x
    
    
class BranchedCoughCNN(nn.Module):
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        
        self.spec_branch, self.spec_skips = _branch_and_skips([(3, 32, 2), (32, 64, 2), (64, 128, 1)], 5)
        self.mfcc_branch, self.mfcc_skips = _branch_and_skips([(3, 32, 2), (32, 64, 2), (64, 128, 1)], 5)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.drop2d: nn.Dropout2d = nn.Dropout2d(0.1)
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU6(),
            nn.Linear(512, num_classes)
        )

    def forward(self, mel: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:

        mel_out = _res_forward(self.spec_branch, self.spec_skips, mel, self.drop2d)   # Shape: [B, 128, H_spec, W_spec]
        mfcc_out = _res_forward(self.mfcc_branch, self.mfcc_skips, mfcc, self.drop2d)  # Shape: [B, 128, H_mfcc, W_mfcc]
        
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