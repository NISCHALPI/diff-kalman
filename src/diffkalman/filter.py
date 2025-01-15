"""
This module provides a differentiable Kalman Filter implementation designed for linear and non-linear dynamical systems perturbed by Gaussian noise. 

The Kalman Filter is implemented using the PyTorch library, allowing seamless integration with neural networks to parameterize the dynamics, observation, and noise models. These parameters are optimized using Stochastic Variational Inference (SVI), which is mathematically equivalent to the classical Expectation-Maximization (EM) algorithm. The Gaussian assumption ensures that the posterior distribution of states remains analytically tractable using Rauch-Tung-Striebel smoother equations. This tractability facilitates the maximization of the log-likelihood of the observed data, given the model parameters.

Key Features:
- Differentiable Kalman Filter implementation.
- Flexible state transition and observation models (linear/non-linear, learnable, or fixed).
- Automatic computation of Jacobians using PyTorch's autograd functionality.
- Support for Monte Carlo sampling to evaluate the expected joint log-likelihood.

Classes:
    DifferentiableKalmanFilter: Implements the core Kalman Filter algorithm for linear and non-linear dynamical systems.

Functions:
    joint_jacobian_transform: Computes joint Jacobians for state and observation functions.
    log_likelihood: Computes the log-likelihood for Gaussian distributions.
    gaussian_log_likelihood: Computes Gaussian log-likelihoods for given inputs.

Example:
    # Define custom state transition and observation functions
    f = MyStateTransitionFunction()
    h = MyObservationFunction()

    # Initialize the Kalman filter
    kalman_filter = DifferentiableKalmanFilter(dim_x=4, dim_z=2, f=f, h=h)

    # Perform filtering on a sequence of observations
    results = kalman_filter.sequence_filter(
        z_seq=observations,
        x0=initial_state,
        P0=initial_covariance,
        Q=process_noise_covariance,
        R=observation_noise_covariance,
    )

Dependencies:
    - PyTorch: For deep learning integration and differentiability.
    - Custom Modules: `joint_jacobian_transform`, `negative_log_likelihood`.

__all__ = ["DifferentiableKalmanFilter"]
"""

import torch
import torch.nn as nn
from .joint_jacobian_transform import joint_jacobian_transform
from .negative_log_likelihood import log_likelihood, gaussain_log_likelihood


__all__ = ["DiffrentiableKalmanFilter"]


class DiffrentiableKalmanFilter(nn.Module):
    """
    Implements a differentiable Kalman Filter for linear and non-linear dynamical systems.

    This class provides methods for prediction, update, and smoothing, enabling filtering of
    sequences of observations while allowing the use of learnable neural network models for
    state transition and observation functions.

    Args:
        dim_x (int): Dimensionality of the state space.
        dim_z (int): Dimensionality of the observation space.
        f (nn.Module): State transition function, which can be linear/non-linear and learnable or fixed.
        h (nn.Module): Observation function, which can be linear/non-linear and learnable or fixed.
        mc_samples (int, optional): Number of Monte Carlo samples used for expected log-likelihood computation. Defaults to 100.

    Attributes:
        dim_x (int): Dimensionality of the state space.
        dim_z (int): Dimensionality of the observation space.
        f (nn.Module): State transition function.
        h (nn.Module): Observation function.
        I (torch.Tensor): Identity matrix of size `(dim_x, dim_x)`.
        _f_joint (Callable): Joint Jacobian function of the state transition function.
        _h_joint (Callable): Joint Jacobian function of the observation function.

    Methods:
        predict: Performs state prediction given the current state and covariance.
        update: Updates the state estimate based on the current observation.
        predict_update: Combines prediction and update steps for a single observation.
        sequence_filter: Filters a sequence of observations to estimate states over time.
        sequence_smooth: Smooths a sequence of observations for refined state estimates.

    Note:
        - PyTorch's autograd functionality is utilized to compute Jacobians of the state transition
          and observation functions, eliminating the need for manual derivation.
        - Monte Carlo sampling is used for approximating the expected log-likelihood in non-linear scenarios.
    """

    def __init__(
        self, dim_x: int, dim_z: int, f: nn.Module, h: nn.Module, mc_samples: int = 100
    ) -> None:
        """Initializes the DiffrentiableKalmanFilter object.

        Args:
            dim_x (int): Dimensionality of the state space.
            dim_z (int): Dimensionality of the observation space.
            f (nn.Module): State transition function which can be parametrized by a neural network which can linear/non-linear and learnable or fixed.
            h (nn.Module): Observation function which can be parametrized by a neural network which can linear/non-linear and learnable or fixed.
            mc_samples (int): Number of Monte Carlo samples to draw to calcualte expected joint log-likelihood. Default is 100.

        Returns:
            None

        Note:
            - The state transition function signature is f(x: torch.Tensor, *args) -> torch.Tensor.
            - The observation function signature is h(x: torch.Tensor, *args) -> torch.Tensor.
        """

        super().__init__()

        # Store the dimensionality of the state and observation space
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Store the state transition and observation functions
        self.f = f
        self.h = h

        # Store the number of Monte Carlo samples
        self.mc_samples = mc_samples

        # Register the Jacobian functions of the state transition and observation functions
        self._f_joint = joint_jacobian_transform(self.f)
        self._h_joint = joint_jacobian_transform(self.h)

        # Identity matrix of shape (dim_x, dim_x)
        self.register_buffer("I", torch.eye(dim_x))

    def predict(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        Q: torch.Tensor,
        f_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """
        Predicts the next state given the current state and the state transition function.

        Args:
            x (torch.Tensor): Current state estimate of shape `(dim_x,)`.
            P (torch.Tensor): Current state covariance of shape `(dim_x, dim_x)`.
            Q (torch.Tensor): Process noise covariance of shape `(dim_x, dim_x)`.
            f_args (tuple, optional): Additional arguments to the state transition function.

        Returns:
            dict[str, torch.Tensor]:
                - `x_prior`: Predicted state estimate of shape `(dim_x,)`.
                - `P_prior`: Predicted state covariance of shape `(dim_x, dim_x)`.
                - `state_jacobian`: Jacobian of the state transition function of shape `(dim_x, dim_x)`.

        Note:
            - The control input, if any, can be incorporated into the state transition function by passing it as an argument in `f_args`.
        """
        # Compute the predicted state estimate and covariance
        F, x_prior = self._f_joint(x, *f_args)

        P_prior = F @ P @ F.T + Q

        return {"x_prior": x_prior, "P_prior": P_prior, "state_jacobian": F}

    def update(
        self,
        z: torch.Tensor,
        x_prior: torch.Tensor,
        P_prior: torch.Tensor,
        R: torch.Tensor,
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """
        Updates the state estimate based on the current observation and observation model.

        Args:
            x (torch.Tensor): Predicted state estimate of shape `(dim_x,)`.
            P (torch.Tensor): Predicted state covariance of shape `(dim_x, dim_x)`.
            z (torch.Tensor): Current observation of shape `(dim_z,)`.
            R (torch.Tensor): Observation noise covariance of shape `(dim_z, dim_z)`.
            h_args (tuple, optional): Additional arguments to the observation function.

        Returns:
            dict[str, torch.Tensor]:
                - `x_post`: Updated state estimate of shape `(dim_x,)`.
                - `P_post`: Updated state covariance of shape `(dim_x, dim_x)`.
                - `y`: Observation residual (innovation) of shape `(dim_z,)`.
                - `S`: Innovation covariance of shape `(dim_z, dim_z)`.
                - `K`: Kalman gain of shape `(dim_x, dim_z)`.

        Note:
            - PyTorch's autograd is used to compute the Jacobian of the observation function, ensuring compatibility with differentiable models.
        """
        # Compute the jacboian of the observation function
        H, z_pred = self._h_joint(x_prior, *h_args)

        # Compute the innovation
        y = z - z_pred
        # Compute the innovation covariance matrix
        S = H @ P_prior @ H.T + R
        # Compute the Kalman gain
        K = P_prior @ H.T @ torch.linalg.inv(S)

        # Update the state vector
        x_post = x_prior + K @ y
        # Update the state covariance matrix using joseph form
        factor = self.I - K @ H
        P_post = factor @ P_prior @ factor.T + K @ R @ K.T

        return {
            "x_post": x_post,
            "P_post": P_post,
            "innovation": y,
            "innovation_covariance": S,
            "observation_jacobian": H,
            "kalman_gain": K,
        }

    def predict_update(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        P: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """
        Combines prediction and update steps for a single observation.

        Args:
            x (torch.Tensor): Current state estimate of shape `(dim_x,)`.
            P (torch.Tensor): Current state covariance of shape `(dim_x, dim_x)`.
            z (torch.Tensor): Current observation of shape `(dim_z,)`.
            Q (torch.Tensor): Process noise covariance of shape `(dim_x, dim_x)`.
            R (torch.Tensor): Observation noise covariance of shape `(dim_z, dim_z)`.
            f_args (tuple, optional): Additional arguments to the state transition function.
            h_args (tuple, optional): Additional arguments to the observation function.

        Returns:
            dict[str, torch.Tensor]:
                - `x_post`: Updated state estimate of shape `(dim_x,)`.
                - `P_post`: Updated state covariance of shape `(dim_x, dim_x)`.
                - `y`: Observation residual (innovation) of shape `(dim_z,)`.
                - `S`: Innovation covariance of shape `(dim_z, dim_z)`.
                - `K`: Kalman gain of shape `(dim_x, dim_z)`.

        Note:
            - This method is particularly useful for real-time filtering where observations arrive sequentially.
        """
        # Predict the next state
        prediction = self.predict(x=x, P=P, Q=Q, f_args=f_args)

        # Update the state estimate
        update = self.update(
            z=z,
            x_prior=prediction["x_prior"],
            P_prior=prediction["P_prior"],
            R=R,
            h_args=h_args,
        )

        return {**prediction, **update}

    def sequence_filter(
        self,
        z_seq: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """
        Filters a sequence of observations to estimate states over time.

        Args:
            z_seq (torch.Tensor): Sequence of observations of shape `(T, dim_z)`, where `T` is the number of time steps.
            x0 (torch.Tensor): Initial state estimate of shape `(dim_x,)`.
            P0 (torch.Tensor): Initial state covariance of shape `(dim_x, dim_x)`.
            Q (torch.Tensor): Process noise covariance of shape `(T, dim_x, dim_x)`.
            R (torch.Tensor): Observation noise covariance of shape `(T, dim_z, dim_z)`.
            f_args (tuple, optional): Additional arguments to the state transition function.
            h_args (tuple, optional): Additional arguments to the observation function.

        Returns:
            dict[str, torch.Tensor]:
                - `x_filtered`: Filtered state estimates of shape `(T, dim_x)`.
                - `P_filtered`: Filtered state covariances of shape `(T, dim_x, dim_x)`.

        Note:
            - Each time step is processed iteratively using the `predict` and `update` methods.
        """
        # Get the sequence length
        T = z_seq.size(0)

        # Initialize the output tensors
        outs_terms = [
            "x_prior",
            "P_prior",
            "x_post",
            "P_post",
            "state_jacobian",
            "innovation",
            "innovation_covariance",
            "observation_jacobian",
            "kalman_gain",
        ]
        outputs = {term: [] for term in outs_terms}

        # Run the Kalman Filter sequentially
        for t in range(T):
            # Predict the next state
            results = self.predict_update(
                z=z_seq[t],
                x=x0 if t == 0 else outputs["x_post"][-1],
                P=P0 if t == 0 else outputs["P_post"][-1],
                Q=Q[t],
                R=R[t],
                f_args=(args[t] for args in f_args),
                h_args=(args[t] for args in h_args),
            )

            # Update the output tensors
            for term in outs_terms:
                outputs[term].append(results[term])

        # Stack the output tensors
        for term in outs_terms:
            outputs[term] = torch.stack(outputs[term])

        # Calculate the log-likelihood
        log_like = log_likelihood(
            innovation=outputs["innovation"],
            innovation_covariance=outputs["innovation_covariance"],
        )
        outputs["log_likelihood"] = log_like

        return outputs

    def marginal_log_likelihood(
        self,
        z_seq: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> torch.Tensor:
        """
        Computes the marginal log-likelihood of the observed data given the model parameters.

        Args:
            z_seq (torch.Tensor): Sequence of observations of shape `(T, dim_z)`, where `T` is the number of time steps.
            x0 (torch.Tensor): Initial state estimate of shape `(dim_x,)`.
            P0 (torch.Tensor): Initial state covariance of shape `(dim_x, dim_x)`.
            Q (torch.Tensor): Process noise covariance of shape `(T, dim_x, dim_x)`.
            R (torch.Tensor): Observation noise covariance of shape `(T, dim_z, dim_z)`.
            f_args (tuple, optional): Additional arguments to the state transition function.
            h_args (tuple, optional): Additional arguments to the observation function.

        Returns:
            torch.Tensor: The marginal log-likelihood of the observed data given the model parameters.
        """
        # Run the Kalman Filter and only track innovation and innovation covariance
        innov = []
        innov_cov = []

        # Initialize the state estimate
        x_post = x0
        P_post = P0

        for t in range(z_seq.size(0)):
            # Predict the next state
            prediction = self.predict(
                x=x_post,
                P=P_post,
                Q=Q[t],
                f_args=(args[t] for args in f_args),
            )

            # Update the state estimate
            update = self.update(
                z=z_seq[t],
                x_prior=prediction["x_prior"],
                P_prior=prediction["P_prior"],
                R=R[t],
                h_args=(args[t] for args in h_args),
            )

            # Update the state estimate
            x_post = update["x_post"]
            P_post = update["P_post"]

            # Store the innovation and innovation covariance
            innov.append(update["innovation"])
            innov_cov.append(update["innovation_covariance"])

        return log_likelihood(
            innovation=torch.stack(innov),
            innovation_covariance=torch.stack(innov_cov),
        )

    def rauch_tung_striebel_smoothing(
        self,
        x0: torch.Tensor,
        P0: torch.Tensor,
        x_prior: torch.Tensor,
        P_prior: torch.Tensor,
        x_filtered: torch.Tensor,
        P_filtered: torch.Tensor,
        state_jacobian: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Smooths a sequence of filtered states for refined estimates over time.

        Args:
            x_filtered (torch.Tensor): Filtered state estimates of shape `(T, dim_x)`.
            P_filtered (torch.Tensor): Filtered state covariances of shape `(T, dim_x, dim_x)`.
            Q (torch.Tensor): Process noise covariance of shape `(dim_x, dim_x)`.
            f_args (tuple, optional): Additional arguments to the state transition function.

        Returns:
            dict[str, torch.Tensor]:
                - `x_smoothed`: Smoothed state estimates of shape `(T, dim_x)`.
                - `P_smoothed`: Smoothed state covariances of shape `(T, dim_x, dim_x)`.

        Note:
            - This method uses Rauch-Tung-Striebel smoothing equations to refine the estimates.
            - Smoothed states incorporate information from future observations, improving the overall estimation accuracy.
        """

        # Initialize the output tensors
        outs_terms = ["x_smoothed", "P_smoothed", "smoothing_gain"]
        outputs = {term: [] for term in outs_terms}

        # Last state estimate is already the smoothed state estimate
        # by definition
        outputs["x_smoothed"].append(x_filtered[-1])
        outputs["P_smoothed"].append(P_filtered[-1])

        # Sequence length
        T = x_filtered.size(0)

        # Start the backward-recursion form the second last time step
        # to the first time step
        for t in range(T - 2, -1, -1):
            # Compute the smoothing gain
            L = (
                P_filtered[t]
                @ state_jacobian[t + 1].T
                @ torch.linalg.inv(P_prior[t + 1])
            )
            # Insert the smoothing gain into the output tensor
            outputs["smoothing_gain"].insert(0, L)
            # Compute the smoothed state estimate
            outputs["x_smoothed"].insert(
                0, x_filtered[t] + L @ (outputs["x_smoothed"][0] - x_prior[t + 1])
            )
            # Compute the smoothed state covariance
            outputs["P_smoothed"].insert(
                0, P_filtered[t] + L @ (outputs["P_smoothed"][0] - P_prior[t + 1]) @ L.T
            )

        # Smoothed the initial state estimate
        L0 = P0 @ state_jacobian[0].T @ torch.linalg.inv(P_prior[0])
        outputs["smoothing_gain"].insert(0, L0)
        x0_smoothed = x0 + L0 @ (outputs["x_smoothed"][0] - x_prior[0])
        P0_smoothed = P0 + L0 @ (outputs["P_smoothed"][0] - P_prior[0]) @ L0.T

        return {
            "x_smoothed": torch.stack(outputs["x_smoothed"]),
            "P_smoothed": torch.stack(outputs["P_smoothed"]),
            "smoothing_gain": torch.stack(outputs["smoothing_gain"]),
            "x0_smoothed": x0_smoothed,
            "P0_smoothed": P0_smoothed,
        }

    def sequence_smooth(
        self,
        z_seq: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> dict[str, torch.Tensor]:
        """
        Applies a smoothing algorithm to a sequence of observations using a two-step process:
        Kalman filtering followed by Rauch-Tung-Striebel smoothing.

        Args:
            z_seq (torch.Tensor): Sequence of observations with shape `(seq_len, dim_z)`.
            x0 (torch.Tensor): Initial state estimate with shape `(dim_x,)`.
            P0 (torch.Tensor): Initial state covariance matrix with shape `(dim_x, dim_x)`.
            Q (torch.Tensor): Process noise covariance matrix with shape `(seq_len, dim_x, dim_x)`.
            R (torch.Tensor): Observation noise covariance matrix with shape `(seq_len, dim_z, dim_z)`.
            f_args (tuple): Optional sequence of additional arguments for the state transition function.
            h_args (tuple): Optional sequence of additional arguments for the observation function.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - Filtered state estimates (`x_post`) and covariances (`P_post`).
                - Smoothed state estimates (`x_smooth`) and covariances (`P_smooth`).

        Notes:
            - The `f_args` and `h_args` sequences correspond to specific time steps.
            - For time-invariant `Q` or `R`, use PyTorch's `repeat` functionality to create sequences.
        """
        # Run the Kalman Filter
        filter_results = self.sequence_filter(
            z_seq=z_seq,
            x0=x0,
            P0=P0,
            Q=Q,
            R=R,
            f_args=f_args,
            h_args=h_args,
        )

        # Run the Rauch-Tung-Striebel Smoothing
        smooth_results = self.rauch_tung_striebel_smoothing(
            x0=x0,
            P0=P0,
            x_prior=filter_results["x_prior"],
            P_prior=filter_results["P_prior"],
            x_filtered=filter_results["x_post"],
            P_filtered=filter_results["P_post"],
            state_jacobian=filter_results["state_jacobian"],
        )

        return {**filter_results, **smooth_results}

    def draw_samples(
        self, n_samples: int, mu: torch.Tensor, P: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates samples from a multivariate normal distribution parameterized
        by a mean vector and covariance matrix.

        Args:
            n_samples (int): Number of samples to generate.
            mu (torch.Tensor): Mean vector of the distribution with shape `(dim_x,)`.
            P (torch.Tensor): Covariance matrix of the distribution with shape `(dim_x, dim_x)`.

        Returns:
            torch.Tensor: A tensor of generated samples with shape `(n_samples, dim_x)`.

        Notes:
            - Cholesky decomposition is used to ensure numerical stability.
            - The samples are drawn using PyTorch's distribution utilities.
        """
        # Draw samples from a multivariate normal distribution
        X_unit_samples = (
            torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.dim_x), covariance_matrix=torch.eye(self.dim_x)
            )
            .sample((n_samples,))
            .to(device=mu.device, dtype=mu.dtype)
        )

        return mu + torch.einsum("kk,jk->jk", torch.linalg.cholesky(P), X_unit_samples)

    def monte_carlo_expected_joint_log_likekihood(
        self,
        z_seq: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        x0_smoothed: torch.Tensor,
        P0_smoothed: torch.Tensor,
        x_smoothed: torch.Tensor,
        P_smoothed: torch.Tensor,
        Q_seq: torch.Tensor,
        R_seq: torch.Tensor,
        f_args: tuple = (),
        h_args: tuple = (),
    ) -> torch.Tensor:
        """
        This function computes the Monte Carlo approximation of the expected joint log-likelihood 
        of the observed data given the posterior distribution of the states.

        The posterior distribution of the states is derived from the smoothed state estimates, 
        which are tractable and exact in the case of linear Gaussian models. This simplifies the 
        process compared to a full-fledged Variational Inference (VI) approach, where the posterior 
        distribution is approximated using a variational distribution. Here, the exact posterior 
        distribution is used, as it is directly obtained from the smoothing algorithm, eliminating 
        the need for variational approximations.

        In the case where the dynamics and observation models are parameterized by neural networks, 
        the variational distribution corresponds to the exact posterior distribution over the unknown 
        variables (e.g., parameters of the models). This formulation maintains computational efficiency 
        while leveraging the expressiveness of neural networks for system modeling.

        Args:
            z_seq (torch.Tensor): Sequence of observations of shape (seq_len, dim_z).\
            x0 (torch.Tensor): Initial state estimate of shape (dim_x,).
            P0 (torch.Tensor): Initial state covariance of shape (dim_x, dim_x).
            x0_smoothed (torch.Tensor): Initial smoothed state estimate of shape (dim_x,).
            P0_smoothed (torch.Tensor): Initial smoothed state covariance of shape (dim_x, dim_x).
            x_smoothed (torch.Tensor): Smoothed state estimates of shape (seq_len, dim_x).
            P_smoothed (torch.Tensor): Smoothed state covariances of shape (seq_len, dim_x, dim_x).
            Q_seq (torch.Tensor): Process noise covariance of shape (seq_len, dim_x, dim_x).
            R_seq (torch.Tensor): Observation noise covariance of shape (seq_len, dim_z, dim_z).
            f_args (tuple): Additional arguments to the state transition function.
            h_args (tuple): Additional arguments to the observation function.

       
        Returns:
            torch.Tensor: The Monte Carlo approximation of the expected joint log-likelihood.
        """
        # Loop to calculate the monte-carlo approximation of the expected joint log-likelihood
        E_log_likelihood = 0.0
        # Here we will use the vmap function to vectorize the computation of the log-likelihood
        vllf = torch.vmap(
            func=gaussain_log_likelihood,
            in_dims=(0, None),
        )

        # Draw samples from the smoothed initial distribution
        X0_samples = self.draw_samples(
            n_samples=self.mc_samples,
            mu=x0_smoothed,
            P=P0_smoothed,
        )
        innov_x0 = X0_samples - x0
        E_log_likelihood += vllf(innov_x0, P0).mean()

        # For each measurement calculate the expected log-likelihood
        for i in range(z_seq.size(0)):
            # Calculate the measurement term
            Xt = self.draw_samples(
                n_samples=self.mc_samples,
                mu=x_smoothed[i],
                P=P_smoothed[i],
            )
            innov_z = z_seq[i] - torch.vmap(
                lambda x: self.h(x, *(args[i] for args in h_args)),
            )(Xt)
            E_log_likelihood += vllf(innov_z, R_seq[i]).mean()

            # Calculate the transition term
            Xp = self.draw_samples(
                n_samples=self.mc_samples,
                mu=x_smoothed[i - 1] if i > 0 else x0_smoothed,
                P=P_smoothed[i - 1] if i > 0 else P0_smoothed,
            )
            innov_x = Xt - torch.vmap(
                lambda x: self.f(x, *(args[i] for args in f_args)),
            )(Xp)

            E_log_likelihood += vllf(innov_x, Q_seq[i]).mean()

        return E_log_likelihood
