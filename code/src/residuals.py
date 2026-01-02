
"""
For every dataset that we want to use equation knowledge, we define the residuals here.
"""

import torch


def unit_sphere_residual(x: torch.Tensor) -> torch.Tensor:
	"""
	Residual function for the n-dimensional unit sphere: r(x) = ||x||^2 - 1
	Args:
		x: Tensor of shape (batch_size, D) representing N points in D dimensions
	Returns:
		Tensor of shape (batch_size,) with residuals for each point
	"""
	residuals = torch.sum(x ** 2, dim=1) - 1
	return residuals


