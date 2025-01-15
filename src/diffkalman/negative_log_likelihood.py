"""
Utility functions for calculating negative log likelihood of observations.

This module provides functions to compute the negative log likelihood of a sequence of observations given 
their predicted values and associated covariance matrices. The negative log likelihood quantifies the 
likelihood of observing the data under a given model, incorporating uncertainties in the measurements.

The negative log likelihood for a sequence of innovations and their covariances is computed using the 
logarithm of the joint distribution of the observations given the parameters. It is expressed mathematically as:

$$
\ln(\mathcal{L}(\Phi \mid \mathcal{D})) = -\frac{nk}{2} \ln(2\pi) - \frac{1}{2} \sum_{i=1}^{n} \left[ \ln |S_{i}| + \Delta y_{i}^T S_{i}^{-1} \Delta y_i \right]
$$

where $ n $ is the sequence length, $ k $ is the dimension of the innovation vector, 
$ \Delta y_i $ is the innovation vector at time step $ i $, and $ S_i $ is the innovation 
covariance matrix at time step $ i $.

Functions:
- `likelihood_at_time`: Calculates likelihood at a single time step.
- `log_likelihood`: Computes total log likelihood for a sequence of time steps.
- `negative_log_likelihood`: Computes total negative log likelihood for a sequence of time steps.

Note:
- The constant factor $ \frac{nk}{2} \ln(2\pi) $ is skipped in the implementation since it does not affect the optimization process.
"""

import torch

__all__ = ["gaussain_log_likelihood", "log_likelihood"]


def gaussain_log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the likelihood estimation at a single time step.

    Args:
        innovation (torch.Tensor): The innovation vector, representing the difference
                                   between the observed and predicted measurements.
                                   Shape: (dim_z,)
        innovation_covariance (torch.Tensor): The innovation covariance matrix,
                                              representing the uncertainty in the innovation.
                                              Shape: (dim_z, dim_z)

    Returns:
        torch.Tensor: The loss value for the given innovation and its covariance.

    Note:
        - The dimension factor is skipped since it is a constant and does not affect the optimization.

    Example:
        >>> innovation = torch.tensor([1.0, 2.0])
        >>> innovation_covariance = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> loss = likelihood_at_time(innovation, innovation_covariance)
        >>> print(loss)
    """
    # Calculate the log determinant of the innovation covariance matrix
    log_det = torch.linalg.slogdet(innovation_covariance)[1]

    k = innovation.shape[0]
    C = (-k / 2) * torch.log(
        2 * torch.tensor(torch.pi, dtype=innovation.dtype, device=innovation.device)
    )
    return C - 0.5 * (
        log_det + innovation @ torch.linalg.inv(innovation_covariance) @ innovation
    )


def log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the log likelihood estimation for a sequence of time steps.

    Args:
        innovation (torch.Tensor): The innovation vector for each time step,
                                   representing the differences between observed and predicted
                                   measurements over a sequence. Shape: (seq_len, dim_z)
        innovation_covariance (torch.Tensor): The innovation covariance matrix for each time step,
                                              representing the uncertainty in the innovations
                                              over a sequence. Shape: (seq_len, dim_z, dim_z)

    Returns:
        torch.Tensor: The total log likelihood value for the sequence of innovations and
                      their covariances.

    Example:
        >>> innovation = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        >>> innovation_covariance = torch.tensor([
        >>>     [[1.0, 0.0], [0.0, 1.0]],
        >>>     [[0.5, 0.0], [0.0, 0.5]]
        >>> ])
        >>> total_log_likelihood = log_likelihood(innovation, innovation_covariance)
        >>> print(total_log_likelihood)
    """
    return torch.vmap(gaussain_log_likelihood, in_dims=0)(
        innovation, innovation_covariance
    ).sum()


def negative_log_likelihood(
    innovation: torch.Tensor,
    innovation_covariance: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the negative log likelihood estimation for a sequence of time steps.

    Args:
        innovation (torch.Tensor): The innovation vector for each time step,
                                   representing the differences between observed and predicted
                                   measurements over a sequence. Shape: (seq_len, dim_z)
        innovation_covariance (torch.Tensor): The innovation covariance matrix for each time step,
                                              representing the uncertainty in the innovations
                                              over a sequence. Shape: (seq_len, dim_z, dim_z)

    Returns:
        torch.Tensor: The total negative log likelihood value for the sequence of innovations and
                      their covariances.

    Example:
        >>> innovation = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        >>> innovation_covariance = torch.tensor([
        >>>     [[1.0, 0.0], [0.0, 1.0]],
        >>>     [[0.5, 0.0], [0.0, 0.5]]
        >>> ])
        >>> total_negative_log_likelihood = negative_log_likelihood(innovation, innovation_covariance)
        >>> print(total_negative_log_likelihood)
    """
    return -log_likelihood(innovation, innovation_covariance)
