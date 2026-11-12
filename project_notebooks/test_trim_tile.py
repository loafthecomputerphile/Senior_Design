"""test_trim_tile: trim trailing zero-pad + tile to full window (plain asserts)."""
import numpy as np
import torch

from modules.preprocessing.audio_process import trim_tile

# short signal trailing-padded: [1,2,3,0,0,0,0,0] -> [1,2,3,1,2,3,1,2]
x = torch.tensor([[1., 2., 3., 0., 0., 0., 0., 0.]])
out = trim_tile(x)
assert out.shape == (1, 8)
assert torch.equal(out, torch.tensor([[1., 2., 3., 1., 2., 3., 1., 2.]])), out

# full-length signal unchanged
full = torch.rand(1, 16)
assert torch.equal(trim_tile(full), full)

# all-zero row stays all-zero (no div/argmax crash)
z = trim_tile(torch.zeros(1, 8))
assert torch.equal(z, torch.zeros(1, 8))

# 1-D input round-trips
one_d = trim_tile(torch.tensor([1., 0., 0., 0., 0.]))
assert one_d.shape == (5,) and torch.equal(one_d, torch.tensor([1., 1., 1., 1., 1.]))

# explicit target length: 3-sample cough tiled to 16000
seg = torch.tensor([[1., 2., 3.]])
assert trim_tile(seg, n=16000).shape == (1, 16000)
assert torch.equal(trim_tile(seg, n=7), torch.tensor([[1., 2., 3., 1., 2., 3., 1.]]))

# batch works and preserves dtype/device semantics
b = torch.tensor([[1., 2., 0., 0.], [0., 0., 0., 0.]])
ob = trim_tile(b)
assert ob.shape == (2, 4) and torch.equal(ob[0], torch.tensor([1., 2., 1., 2.])) and ob[1].abs().sum() == 0

print("OK: trim_tile all checks passed")
