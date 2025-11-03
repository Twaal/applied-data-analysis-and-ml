"""
Reusable regression utilities and gradient descent routines for Project 2.

This consolidates functions and classes originally developed in Project 1 so
they can be imported in `project2.ipynb` as baselines/benchmarks for neural nets.

Contents
- Feature utilities: polynomial_features
- Closed-form regression: OLS_parameters, Ridge_parameters
- Metrics: MSE, R2
- Optimizers and training loops:
  * gradient: analytic gradient for OLS/Ridge
  * Optimizer: plain GD, Momentum, RMSProp, Adam
  * fit_full_batch: full-batch GD
  * fit_sgd: stochastic (mini-batch) GD with replacement

All functions are NumPy-based and do not depend on scikit-learn for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# -------------------------------
# Feature utilities
# -------------------------------

def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
	"""Construct polynomial features up to given degree (inclusive).

	Inputs
	- x: shape (n,) or (n,)
	- degree: non-negative integer

	Output
	- X: shape (n, degree+1) with columns [x^0, x^1, ..., x^degree]
	"""
	x = np.asarray(x).reshape(-1)
	n = x.shape[0]
	X = np.zeros((n, degree + 1), dtype=float)
	for i in range(degree + 1):
		X[:, i] = x**i
	return X


# -------------------------------
# Closed-form regression
# -------------------------------

def OLS_parameters(X: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Closed-form OLS: (X^T X)^(-1) X^T y

	Note: Mirrors the Project 1 implementation which used np.linalg.inv.
	For improved numerical stability in new work, consider np.linalg.pinv.
	"""
	return np.linalg.inv(X.T @ X) @ (X.T @ y)


def Ridge_parameters(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
	"""Closed-form Ridge: (X^T X + lam I)^(-1) X^T y

	Does NOT regularize the bias term (first coefficient). This matches common
	practice; set lam_bias=True to also regularize bias if desired.
	"""
	n_features = X.shape[1]
	I = np.eye(n_features)
	# Do not regularize intercept (bias) term in column 0
	I[0, 0] = 0.0
	return np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)


# -------------------------------
# Metrics and helpers
# -------------------------------

def runge(x: np.ndarray) -> np.ndarray:
	"""Runge function used in Project 1 synthetic data."""
	x = np.asarray(x)
	return 1.0 / (1.0 + 25.0 * x**2)


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	n = y_true.size
	return float(np.sum((y_true - y_pred) ** 2) / n)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	return float(1.0 - ss_res / ss_tot)


# -------------------------------
# Gradients and optimizers
# -------------------------------

def gradient(
	theta: np.ndarray,
	X: np.ndarray,
	y: np.ndarray,
	lam: float = 0.0,
	ridge: bool = False,
) -> np.ndarray:
	"""Gradient of MSE loss for linear model with optional Ridge.

	Loss(OLS)  = (1/n) ||y - X theta||^2
	Loss(Ridge)= (1/n) ||y - X theta||^2 + lam * ||theta||^2 (no bias reg)
	"""
	n = X.shape[0]
	residual = X @ theta - y
	g = (2.0 / n) * (X.T @ residual)
	if ridge and lam > 0.0:
		reg = 2.0 * lam * theta
		reg[0] = 0.0  # do not regularize bias term
		g = g + reg
	return g


@dataclass
class Optimizer:
	"""Simple optimizer supporting 'gd', 'momentum', 'rmsprop', 'adam'."""

	name: str = "gd"
	eta: float = 1e-2
	# Momentum
	mu: float = 0.9
	# RMSProp
	rho: float = 0.9
	# Adam
	beta1: float = 0.9
	beta2: float = 0.999
	eps: float = 1e-8

	# Internal state (initialized on first step)
	v: Optional[np.ndarray] = None  # for momentum / Adam (m)
	s: Optional[np.ndarray] = None  # for RMSProp / Adam (v)
	t: int = 0  # Adam timestep

	def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
		name = self.name.lower()
		if name == "gd":
			return theta - self.eta * g

		if name == "momentum":
			if self.v is None:
				self.v = np.zeros_like(theta)
			self.v = self.mu * self.v - self.eta * g
			return theta + self.v

		if name == "rmsprop":
			if self.s is None:
				self.s = np.zeros_like(theta)
			self.s = self.rho * self.s + (1.0 - self.rho) * (g * g)
			return theta - self.eta * g / (np.sqrt(self.s) + self.eps)

		if name == "adam":
			if self.v is None:
				self.v = np.zeros_like(theta)
			if self.s is None:
				self.s = np.zeros_like(theta)
			self.t += 1
			# m, v estimates
			self.v = self.beta1 * self.v + (1.0 - self.beta1) * g
			self.s = self.beta2 * self.s + (1.0 - self.beta2) * (g * g)
			# bias corrections
			m_hat = self.v / (1.0 - self.beta1**self.t)
			v_hat = self.s / (1.0 - self.beta2**self.t)
			return theta - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)

		# Fallback to plain GD
		return theta - self.eta * g


# -------------------------------
# Training loops
# -------------------------------

def _init_theta(n_features: int, seed: Optional[int] = None) -> np.ndarray:
	rng = np.random.default_rng(seed)
	return rng.normal(loc=0.0, scale=1e-2, size=(n_features,))


def fit_full_batch(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	ridge_lambda: float,
	optimizer: Optimizer,
	ridge: bool = False,
	max_epochs: int = 2000,
	tol: float = 1e-8,
	init_theta: Optional[np.ndarray] = None,
	seed: Optional[int] = None,
) -> Dict[str, object]:
	"""Full-batch gradient descent training loop.

	Returns a dict with keys: theta, mse_train, mse_test, epoch, history
	"""
	n_features = X_train.shape[1]
	theta = _init_theta(n_features, seed) if init_theta is None else init_theta.copy()

	history: List[float] = []
	for epoch in range(1, max_epochs + 1):
		g = gradient(theta, X_train, y_train, lam=ridge_lambda, ridge=ridge)
		new_theta = optimizer.step(theta, g)
		# Convergence check
		if np.linalg.norm(new_theta - theta) < tol:
			theta = new_theta
			break
		theta = new_theta

		if epoch % 10 == 0 or epoch == 1:
			ytr = X_train @ theta
			history.append(MSE(y_train, ytr))

	ytr = X_train @ theta
	yte = X_test @ theta
	return {
		"theta": theta,
		"mse_train": MSE(y_train, ytr),
		"mse_test": MSE(y_test, yte),
		"epoch": epoch,
		"history": history,
	}


def fit_sgd(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	ridge_lambda: float,
	optimizer: Optimizer,
	ridge: bool = False,
	epochs: int = 100,
	batch_size: int = 32,
	shuffle: bool = True,
	replacement: bool = True,
	init_theta: Optional[np.ndarray] = None,
	seed: Optional[int] = None,
) -> Dict[str, object]:
	"""Stochastic (mini-batch) gradient descent training loop.

	Draws each mini-batch independently with replacement by default to match
	the Project 1 variant noted in comments.

	Returns a dict with keys: theta, mse_train, mse_test, epoch, history
	"""
	rng = np.random.default_rng(seed)
	n_samples, n_features = X_train.shape
	theta = _init_theta(n_features, seed) if init_theta is None else init_theta.copy()

	history: List[float] = []
	steps_per_epoch = max(1, int(np.ceil(n_samples / batch_size)))

	for epoch in range(1, epochs + 1):
		if shuffle and not replacement:
			indices = rng.permutation(n_samples)
		else:
			indices = np.arange(n_samples)

		for step in range(steps_per_epoch):
			if replacement:
				batch_idx = rng.integers(0, n_samples, size=(batch_size,))
			else:
				start = step * batch_size
				end = min(n_samples, (step + 1) * batch_size)
				batch_idx = indices[start:end]

			Xb = X_train[batch_idx]
			yb = y_train[batch_idx]
			g = gradient(theta, Xb, yb, lam=ridge_lambda, ridge=ridge)
			theta = optimizer.step(theta, g)

		if epoch % 5 == 0 or epoch == 1:
			ytr = X_train @ theta
			history.append(MSE(y_train, ytr))

	ytr = X_train @ theta
	yte = X_test @ theta
	return {
		"theta": theta,
		"mse_train": MSE(y_train, ytr),
		"mse_test": MSE(y_test, yte),
		"epoch": epoch,
		"history": history,
	}


__all__ = [
	# utils
	"polynomial_features",
	"OLS_parameters",
	"Ridge_parameters",
	"runge",
	"MSE",
	"R2",
	# optimization
	"gradient",
	"Optimizer",
	"fit_full_batch",
	"fit_sgd",
]

