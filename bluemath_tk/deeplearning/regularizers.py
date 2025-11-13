"""
Project: BlueMath_tk
Sub-Module: deeplearning.regularizers
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Regularization functions for PyTorch models.
"""

import torch


def orthogonal_regularizer(W: torch.Tensor, strength: float = 1e-3) -> torch.Tensor:
    """
    Weight orthogonality regularizer.

    Encourages the weight matrix W to be orthogonal by penalizing
    deviations of W^T W from the identity matrix.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix, shape (out_features, in_features).
    strength : float, optional
        Strength of the orthogonality penalty, by default 1e-3.

    Returns
    -------
    torch.Tensor
        Scalar penalty value.

    Examples
    --------
    >>> import torch
    >>> from bluemath_tk.deeplearning.regularizers import orthogonal_regularizer
    >>> W = torch.randn(20, 128)
    >>> penalty = orthogonal_regularizer(W, strength=1e-3)
    """

    # W: shape (out_features, in_features). We want W^T W ≈ I_k
    # For orthogonality, we typically want W W^T ≈ I (for square) or W^T W ≈ I
    # Assuming W is (k, in_dim), we want W^T W ≈ I_k
    WT_W = torch.matmul(W, W.t())  # (k, k)
    I_k = torch.eye(WT_W.size(0), device=WT_W.device, dtype=WT_W.dtype)

    return strength * torch.sum((WT_W - I_k) ** 2)


def l2_regularizer(parameters, strength: float = 1e-4) -> torch.Tensor:
    """
    L2 regularization (weight decay).

    Parameters
    ----------
    parameters : iterable of torch.Tensor
        Model parameters to regularize.
    strength : float, optional
        Strength of the L2 penalty, by default 1e-4.

    Returns
    -------
    torch.Tensor
        Scalar penalty value.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from bluemath_tk.deeplearning.regularizers import l2_regularizer
    >>> model = nn.Linear(10, 5)
    >>> penalty = l2_regularizer(model.parameters(), strength=1e-4)
    """

    l2_loss = 0.0
    for param in parameters:
        l2_loss += torch.sum(param**2)

    return strength * l2_loss


def l1_regularizer(parameters, strength: float = 1e-4) -> torch.Tensor:
    """
    L1 regularization (sparsity).

    Parameters
    ----------
    parameters : iterable of torch.Tensor
        Model parameters to regularize.
    strength : float, optional
        Strength of the L1 penalty, by default 1e-4.

    Returns
    -------
    torch.Tensor
        Scalar penalty value.

    Examples
    --------
    >>> import torch.nn as nn
    >>> from bluemath_tk.deeplearning.regularizers import l1_regularizer
    >>> model = nn.Linear(10, 5)
    >>> penalty = l1_regularizer(model.parameters(), strength=1e-4)
    """

    l1_loss = 0.0
    for param in parameters:
        l1_loss += torch.sum(torch.abs(param))

    return strength * l1_loss
