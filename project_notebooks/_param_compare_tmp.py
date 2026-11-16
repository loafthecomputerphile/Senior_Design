import sys, io, contextlib
sys.path.insert(0, 'modules')
import torch.nn as nn
import modules.nn.base_nn as B

# expansion=4 (current)
m4 = B.DefaultCNN(3)
p4 = sum(p.numel() for p in m4.parameters())

# expansion=1 == the old block (no widening): emulate by zeroing expansion effect
old_block = B.DepthwiseSeparableBlock.__init__
def patched(self, in_channels, out_channels, stride=1):
    old_block(self, in_channels, out_channels, stride, expansion=1)
B.DepthwiseSeparableBlock.__init__ = patched
m1 = B.DefaultCNN(3)
p1 = sum(p.numel() for p in m1.parameters())

print(f'expansion=4: {p4:,} params')
print(f'expansion=1 (old block): {p1:,} params')
