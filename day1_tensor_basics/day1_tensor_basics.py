import torch
import numpy as np

# Create Tensors
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.randn((3, 3))
d = torch.ones((2, 2), dtype=torch.float32)

# Operations
print(a + 2)
print(c @ c.T)  # Matrix multiplication
print(c.mean(), c.std())

# Reshape and slicing
print(c.shape)
print(c[0])
print(c.view(-1))  # Flatten

# Conversion
np_arr = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_arr)
print(tensor_from_np)

# GPU usage (if available)
if torch.cuda.is_available():
    c = c.cuda()
    print("Tensor moved to GPU:", c)
