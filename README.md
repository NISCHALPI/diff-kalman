# Differentiable Kalman Filter

A PyTorch-based implementation of a differentiable Kalman Filter designed for both linear and non-linear dynamical systems with Gaussian noise. This module seamlessly integrates with neural networks, enabling learnable dynamics, observation, and noise models optimized through Stochastic Variational Inference (SVI).

## Features

- **Fully Differentiable**: End-to-end differentiable implementation compatible with PyTorch's autograd
- **Flexible Models**: Support for both linear and non-linear state transition and observation models
- **Neural Network Integration**: Models can be parameterized using neural networks
- **Automatic Jacobian Computation**: Utilizes PyTorch's autograd for derivative calculations
- **Monte Carlo Sampling**: Supports evaluation of expected joint log-likelihood to perform Expectation-Maximization (EM) learning
- **Rauch-Tung-Striebel Smoothing**: Implements forward-backward smoothing for improved state estimation using RTS algorithm

## Installation

```bash
pip install torch  # Required dependency
# Add your package installation command here
```

## Quick Start

Here's a simple example of using the Differentiable Kalman Filter:

```python
import torch
from diffkalman import DifferentiableKalmanFilter
from diffkalman.utils import SymmetricPositiveDefiniteMatrix
from diffkalman.em_loop import em_updates

# Define custom state transition and observation functions
class StateTransition(torch.nn.Module):
    def forward(self, x, *args):
        # Your state transition logic here
        return x

class ObservationModel(torch.nn.Module):
    def forward(self, x, *args):
        # Your observation logic here
        return x

# Initialize the filter
f = StateTransition()
h = ObservationModel()
Q = SymmetricPositiveDefiniteMatrix(dim=4, trainable=True)
R = SymmetricPositiveDefiniteMatrix(dim=2, trainable=True)
kalman_filter = DifferentiableKalmanFilter(
    dim_x=4,  # State dimension
    dim_z=2,  # Observation dimension
    f=f,      # State transition function
    h=h       # Observation function
)

# Run the filter
results = kalman_filter.sequence_filter(
    z_seq=observations,           # Shape: (T, dim_z)
    x0=initial_state,            # Shape: (dim_x,)
    P0=initial_covariance,       # Shape: (dim_x, dim_x)
    Q=Q().repeat(len(observations), 1, 1),  # Shape: (T, dim_x, dim_x)
    R=R().repeat(len(observations), 1, 1)   # Shape: (T, dim_z, dim_z)
)
```

## Detailed Usage

### State Estimation

The module provides three main estimation methods:

1. **Filtering**: Forward pass only
```python
filtered_results = kalman_filter.sequence_filter(
    z_seq=observations,
    x0=initial_state,
    P0=initial_covariance,
    Q=process_noise,
    R=observation_noise
)
```

2. **Smoothing**: Forward-backward pass
```python
smoothed_results = kalman_filter.sequence_smooth(
    z_seq=observations,
    x0=initial_state,
    P0=initial_covariance,
    Q=process_noise,
    R=observation_noise
)
```

3. **Single-step Prediction**: For real-time applications
```python
step_result = kalman_filter.predict_update(
    z=current_observation,
    x=current_state,
    P=current_covariance,
    Q=process_noise,
    R=observation_noise
)
```

### Parameter Learning

The module supports learning model parameters through using backpropagation using the negative expected joint log-likelihood of the 
data as the loss function.

```python
# Define optimizer
optimizer = torch.optim.Adam(params=[
    {'params': kalman_filter.f.parameters()},
    {'params': kalman_filter.h.parameters()},
    {'params': Q.parameters()},
    {'params': R.parameters()}
]

NUM_EPOCHS = 10
NUM_CYCLES = 10

# Run the EM loop
marginal_likelihoods = em_updates(
    kalman_filter=kalman_filter,
    z_seq=observations,
    x0=initial_state,
    P0=initial_covariance,
    Q=Q,
    R=R,
    optimizer=optimizer,
    num_cycles=NUM_CYCLES,
    num_epochs=NUM_EPOCHS
) 
    
```

## API Reference

### DifferentiableKalmanFilter

Main class implementing the Kalman Filter algorithm.

#### Constructor Parameters:
- `dim_x` (int): State space dimension
- `dim_z` (int): Observation space dimension
- `f` (nn.Module): State transition function
- `h` (nn.Module): Observation function
- `mc_samples` (int, optional): Number of Monte Carlo samples for log-likelihood estimation

#### Key Methods:
- `predict`: State prediction step
- `update`: Measurement update step
- `predict_update`: Combined prediction and update
- `sequence_filter`: Full sequence filtering
- `sequence_smooth`: Full sequence smoothing
- `marginal_log_likelihood`: Compute marginal log-likelihood
- `monte_carlo_expected_joint_log_likekihood`: Estimate expected joint log-likelihood

## Requirements

- PyTorch >= 1.9.0
- Python >= 3.7

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
