import torch
import torch.nn as nn
import torch.optim as optim


class _RBFKernel(nn.Module):
    def __init__(self, gamma=None):
        super(_RBFKernel, self).__init__()
        self._set_gamma(gamma)  # Initialize gamma parameter

    def _set_gamma(self, gamma):
        if gamma is None:
            gamma = 1.0  # Default value for gamma
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.float)
        )  # Set gamma as a learnable parameter

    def forward(self, x, y):
        dist = torch.cdist(x, y) ** 2  # Compute squared Euclidean distance
        return torch.exp(-self.gamma * dist)  # Apply RBF kernel formula


class _PolynomialKernel(nn.Module):
    def __init__(self, degree=3, coef0=1, alpha=1):
        super(_PolynomialKernel, self).__init__()
        self.degree = degree  # Degree is a fixed hyperparameter
        self.coef0 = nn.Parameter(
            torch.tensor(coef0, dtype=torch.float)
        )  # Coefficient for the polynomial term
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float)
        )  # Scaling factor for the dot product

    def forward(self, x, y):
        # Compute dot product between x and y, and scale by alpha
        product = self.alpha * torch.bmm(x, y.transpose(1, 2))
        # Add coef0 to the product
        sum_with_coef = product + self.coef0
        # Compute the power of 'degree'
        result = sum_with_coef**self.degree
        return result


class _SigmoidKernel(nn.Module):
    def __init__(self, alpha=1.0, c=0.0):
        super(_SigmoidKernel, self).__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float)
        )  # Scaling factor for the dot product
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float))  # Bias term

    def forward(self, x, y):
        # Apply sigmoid kernel formula
        return torch.tanh(self.alpha * torch.bmm(x, y.transpose(1, 2)) + self.c)


class _LaplacianKernel(nn.Module):
    def __init__(self, gamma=None):
        super(_LaplacianKernel, self).__init__()
        self._set_gamma(gamma)  # Initialize gamma parameter

    def _set_gamma(self, gamma):
        if gamma is None:
            gamma = 1.0  # Default value for gamma
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.float)
        )  # Set gamma as a learnable parameter

    def forward(self, x, y):
        dist = torch.cdist(x, y)  # Compute Manhattan distance
        return torch.exp(-self.gamma * dist)  # Apply Laplacian kernel formula


class PairwiseKernels(nn.Module):
    def __init__(self):
        super(PairwiseKernels, self).__init__()
        self._initialize_kernels()  # Initialize all kernel types

    def _initialize_kernels(self):
        self.kernels = nn.ModuleDict(
            {
                "rbf": _RBFKernel(),
                "polynomial": _PolynomialKernel(),
                "poly": _PolynomialKernel(),
                "sigmoid": _SigmoidKernel(),
                "sig": _SigmoidKernel(),
                "laplacian": _LaplacianKernel(),
                "lap": _LaplacianKernel(),
            }
        )

    def forward(self, X, Y=None, metric="rbf", **kwds):
        if Y is None:
            Y = X  # Use X as Y if Y is not provided

        if metric in self.kernels:
            return self.kernels[metric](X, Y, **kwds)  # Apply selected kernel
        else:
            supported_metrics = ", ".join(self.kernels.keys())
            raise ValueError(
                f"Unknown metric {metric}. Supported metrics are: {supported_metrics}"
            )  # Error for unsupported metrics

    def get_supported_kernels(self):
        return list(self.kernels.keys())
