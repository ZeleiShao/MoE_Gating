import torch
import time

mask = torch.randn(4096, 256) #4.5ms
t1 = time.time()
for i in range(100):
   locations = torch.cumsum(mask, dim=0) - 1
t2 = time.time()
print((t2 - t1) / 100)