# Torch Kernel

This repository contains implementations of various kernel functions for PyTorch. These kernels can be used in machine learning algorithms to measure the similarity between data points. The available kernels include RBF, Polynomial, Sigmoid, and Laplacian, among others.

```python
import torch
from torch_kernel import PairwiseKernels

# Create a kernel
kernel = PairwiseKernels()

# Use the kernel to compute the similarity between two data points
X = torch.tensor([1, 2, 3])
Y = torch.tensor([1, 2, 3])
similarity = model(X, Y, metric="rbf")
print(similarity)
```