"""The EM loop module that implements the EM algorithm for the Differentiable Kalman Filter."""

from .filter import DiffrentiableKalmanFilter
from .utils import SymmetricPositiveDefiniteMatrix
import torch
import torch.nn as nn


__all__ = [
    "em_updates",
]


def em_updates(
    dkf: DiffrentiableKalmanFilter,
    z_seq: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
    Q: SymmetricPositiveDefiniteMatrix,
    R: SymmetricPositiveDefiniteMatrix,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_cycles: int = 20,
    num_epochs: int = 100,
    h_args: tuple = (),
    f_args: tuple = (),
) -> dict:
    """A sample implementation of the EM algorithm for the Differentiable Kalman Filter.

    Args:
        z (torch.Tensor): The noisy measurements sequence. Dimension: (seq_len, obs_dim)
        x0 (torch.Tensor): The initial state vector. Dimension: (state_dim,)
        P0 (torch.Tensor): The initial covariance matrix of the state vector. Dimension: (state_dim, state_dim)
        Q (SymmetricPositiveDefiniteMatrix): The process noise covariance matrix module.
        R (SymmetricPositiveDefiniteMatrix): The measurement noise covariance matrix module.
        optimizer (torch.optim.Optimizer): The optimizer
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler
        num_cycles (int): The number of cycles. Default: 20
        num_epochs (int): The number of epochs. Default: 100
        h_args (tuple): Additional arguments for the measurement model. Default: ()
        f_args (tuple): Additional arguments for the transition function. Default: ()

    Returns:
        dict: log likelihoods of the model with respect to the number of epochs and cycles

    Note:
        - Use the SymmetricPositiveDefiniteMatrix module to ensure that the process noise covariance matrix Q and the measurement noise covariance matrix R are symmetric and positive definite.
        - The optimizer and the learning rate scheduler should be initialized before calling this function.
        - The measurement model and the transition function should be differentiable torch modules.
    """
    likelihoods = torch.zeros(num_epochs, num_cycles)

    # Perform EM updates for num_epochs
    for e in range(num_epochs):

        ## The E-step
        ## Without gradients tracking, get the posterior state distribution wrt current values of the parameters
        with torch.no_grad():
            posterior = dkf.sequence_smooth(
                z_seq=z_seq,
                x0=x0,
                P0=P0,
                Q=Q().repeat(len(z_seq), 1, 1),
                R=R().repeat(len(z_seq), 1, 1),
                f_args=f_args,
                h_args=h_args,
            )

        ## The M-step (Update the parameters) with respect to the current posterior state distribution for num_cycles
        for c in range(num_cycles):
            # Zero the gradients
            optimizer.zero_grad()

            # Compute the marginal likelihood for logging
            with torch.no_grad():
                marginal_likelihood = dkf.marginal_log_likelihood(
                    z_seq=z_seq,
                    x0=x0,
                    P0=P0,
                    Q=Q().repeat(len(z_seq), 1, 1),
                    R=R().repeat(len(z_seq), 1, 1),
                    f_args=f_args,
                    h_args=h_args,
                )
                likelihoods[e, c] = marginal_likelihood

            # Perform the forward pass i.e compute the expected complete joint log-likelihood with respect previous posterior state distribution and current parameters
            complete_log_likelihood = dkf.monte_carlo_expected_joint_log_likekihood(
                z_seq=z_seq,
                x0=x0,
                P0=P0,
                # below represents the posterior state distribution
                x0_smoothed=posterior["x0_smoothed"],
                P0_smoothed=posterior["P0_smoothed"],
                x_smoothed=posterior["x_smoothed"],
                P_smoothed=posterior["P_smoothed"],
                Q_seq=Q().repeat(len(z_seq), 1, 1),
                R_seq=R().repeat(len(z_seq), 1, 1),
                f_args=f_args,
                h_args=h_args,
            )

            # Update the parameters
            (-complete_log_likelihood).backward()
            optimizer.step()
            

            # Print the log likelihood
            print(
                f"Epoch {e + 1}/{num_epochs} Cycle {c + 1}/{num_cycles} Log Likelihood: {marginal_likelihood.item()}"
            )
        
        # Update the learning rate
        lr_scheduler.step()

    return {
        "likelihoods": likelihoods,
    }
