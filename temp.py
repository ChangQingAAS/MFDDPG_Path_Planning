import torch
import numpy as np
tensor = torch.randn(2)
print("tensor:\n", tensor)
array = tensor.numpy()
print("tensor-array:\n", array)
print("array-tensor:\n", torch.from_numpy(array))
