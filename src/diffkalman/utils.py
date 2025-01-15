"""Implementation of utility modules for the differentiable Kalman filter."""

import torch
import torch.nn as nn

__all__ = [
    "SymmetricPositiveDefiniteMatrix",
    "DiagonalSymmetricPositiveDefiniteMatrix",
]


class SymmetricPositiveDefiniteMatrix(nn.Module):
    """
    Module for ensuring a symmetric positive definite matrix using a parameterized approach.

    This module constructs a symmetric positive definite matrix from an initial matrix Q_0.
    It ensures that the resultant matrix is symmetric and has positive eigenvalues.

    Attributes:
        LT_mask (torch.Tensor): Lower triangular mask used for parameter initialization.
        W (nn.Parameter): Parameter representing the input matrix adjusted for symmetry and positivity.

    Methods:
        forward():
            Performs the forward pass of the module.
            Returns a symmetric positive definite matrix derived from the input parameter.
    """

    def __init__(self, M: torch.Tensor, trainable: bool = True):
        """
        Initializes the SymmetricPositiveDefiniteMatrix module.

        Args:
            M (torch.Tensor): Initial matrix for the parameter.
            trainable (bool): Flag to indicate whether the parameter is trainable. Default: True.
        """
        super().__init__()

        # Cholesky decomposition of the initial matrix
        L = torch.linalg.cholesky(M, upper=False)

        # Initialize the parameter with the lower triangular part of the Cholesky decomposition
        self.W = nn.Parameter(L, requires_grad=trainable)

        # Mask for the diagonal entries
        self.diag_mask = torch.eye(M.shape[0], dtype=torch.bool)

    def forward(self) -> torch.Tensor:
        """
        Forward pass of the SymmetricPositiveDefiniteMatrix module.

        Returns:
            torch.Tensor: Symmetric positive definite matrix derived from the input parameter.
        """
        # Make the diagonal entries positive
        L = torch.tril(self.W)
        L[self.diag_mask] = torch.abs(L[self.diag_mask])

        return L @ L.T


class DiagonalSymmetricPositiveDefiniteMatrix(nn.Module):
    """
    A PyTorch module representing a diagonal symmetric positive definite matrix.

    This module takes a diagonal matrix `M` as input and constructs a diagonal symmetric positive definite matrix `W`.
    The diagonal entries of `W` are initialized with the diagonal entries of `M` and can be trained if `trainable` is set to True.

    Args:
        M (torch.Tensor): The diagonal matrix used to initialize the diagonal entries of `W`.
        trainable (bool, optional): Whether the diagonal entries of `W` should be trainable. Defaults to True.

    Returns:
        torch.Tensor: The diagonal symmetric positive definite matrix `W`.
    """

    def __init__(self, M: torch.Tensor, trainable: bool = True):
        super().__init__()

        self.W = nn.Parameter(torch.diag(M.diagonal()), requires_grad=trainable)

    def forward(self) -> torch.Tensor:
        """
        Forward pass of the module.

        Returns:
            torch.Tensor: The diagonal symmetric positive definite matrix `W`.
        """
        # Make the diagonal entries positive
        return torch.abs(self.W)
