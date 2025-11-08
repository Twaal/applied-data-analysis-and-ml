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
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np

# -------------------------------
# Feature utilities
# -------------------------------

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Create a feature matrix X for the features. Here we use polynomial features up to degree 5, plus an intercept column of ones.
def polynomial_features(x, p):
    n = len(x)
    X = np.zeros((n, p + 1))
    for i in range(p + 1):
        X[:, i] = x**i
    return X

def runge_function(x):
    return 1 / (1 + 25 * x**2)

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


# =============================================
# Feed-Forward Neural Network (FFNN) Module
# Migrated from notebook cell to reusable code.
# =============================================

ActivationName = Literal['sigmoid', 'relu', 'leaky_relu', 'linear', 'softmax']
LossName = Literal['mse', 'softmax']  # 'softmax' => categorical cross-entropy with softmax output
RegName = Optional[Literal['l1', 'l2']]
OptName = Literal['gd', 'rmsprop', 'adam']


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
	"""One-hot encode integer labels of shape (n,) into (n, num_classes)."""
	y = y.astype(int).ravel()
	out = np.zeros((y.size, num_classes), dtype=float)
	out[np.arange(y.size), y] = 1.0
	return out


class FFNN:
	"""
	A simple feed-forward neural network implemented in NumPy.

	Essentials:
	- X shape: (n_samples, n_features)
	- y_hat shape: (n_samples, n_outputs)
	- layer_sizes: [n_features, h1, ..., n_outputs]
	- activations per layer: len == len(layer_sizes) - 1
	  Allowed: 'sigmoid', 'relu', 'leaky_relu', 'linear', 'softmax'

	Loss:
	- 'mse': regression; output activation typically 'linear' or 'sigmoid'
	- 'softmax': categorical cross-entropy with softmax output (expects one-hot y)

	Regularization:
	- reg in {None, 'l1', 'l2'} with strength reg_lambda >= 0 on weights (biases not regularized).

	Optimizer:
	- 'gd' (plain), 'rmsprop', 'adam' with typical hyperparameters.

	Metrics:
	- MSE and Accuracy (Accuracy only for softmax classification).

	Initialization:
	- Weights are drawn from a small normal distribution (0.01 * N(0,1)), biases are zeros,
	  matching the course's simple initialization approach.
	"""

	def __init__(
		self,
		layer_sizes: List[int],
		activations: List[ActivationName],
		loss: LossName = 'mse',
		learning_rate: float = 1e-2,
		reg: RegName = None,
		reg_lambda: float = 0.0,
		leaky_slope: float = 1e-2,
		seed: Optional[int] = None,
		# Optimizer config
		optimizer: OptName = 'gd',
		rho: float = 0.9,			# RMSprop decay
		beta1: float = 0.9,		  # Adam first-moment decay
		beta2: float = 0.999,		# Adam second-moment decay
		eps: float = 1e-8,		   # small epsilon for numerical stability
	) -> None:
		assert len(layer_sizes) >= 2, "layer_sizes must have at least input and output"
		assert len(activations) == len(layer_sizes) - 1, "Provide one activation per layer (excluding input)"
		self.layer_sizes = layer_sizes
		self.activations_conf = activations
		self.loss_name: LossName = loss
		self.learning_rate = float(learning_rate)
		self.reg: RegName = reg
		self.reg_lambda = float(reg_lambda)
		self.leaky_slope = float(leaky_slope)
		self.rng = np.random.default_rng(seed)

		# Optimizer
		self.optimizer: OptName = optimizer
		self.rho = float(rho)
		self.beta1 = float(beta1)
		self.beta2 = float(beta2)
		self.eps = float(eps)
		self.t: int = 0  # time step for Adam bias correction

		# Parameters
		self.weights: List[np.ndarray] = []  # (fan_in, fan_out)
		self.biases: List[np.ndarray] = []   # (1, fan_out)

		# Optimizer states
		self._opt_state: Dict[str, List[np.ndarray]] = {}

		# Caches (populated by forward)
		self.zs: List[np.ndarray] = []
		self.as_: List[np.ndarray] = []

		self._initialize_parameters()
		self._initialize_optimizer_state()

	# ---------- Initialization ----------
	def _initialize_parameters(self) -> None:
		self.weights.clear(); self.biases.clear()
		for i in range(len(self.layer_sizes) - 1):
			n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
			# Simple variance-aware scaling by fan-in (course-friendly):
			# - For ReLU/LeakyReLU: std = sqrt(2 / n_in)
			# - For Sigmoid/Linear/Softmax: std = 1 / sqrt(n_in)
			act_name = self.activations_conf[i]
			if act_name in ('relu', 'leaky_relu'):
				std = np.sqrt(2.0 / max(1, n_in))
			else:
				std = 1.0 / np.sqrt(max(1, n_in))
			W = self.rng.standard_normal((n_in, n_out)) * std
			b = np.zeros((1, n_out))  # zero biases
			self.weights.append(W)
			self.biases.append(b)

	def _initialize_optimizer_state(self) -> None:
		self.t = 0
		L = len(self.weights)
		self._opt_state = {}
		if self.optimizer == 'rmsprop':
			self._opt_state['sW'] = [np.zeros_like(self.weights[i]) for i in range(L)]
			self._opt_state['sb'] = [np.zeros_like(self.biases[i]) for i in range(L)]
		elif self.optimizer == 'adam':
			self._opt_state['mW'] = [np.zeros_like(self.weights[i]) for i in range(L)]
			self._opt_state['vW'] = [np.zeros_like(self.weights[i]) for i in range(L)]
			self._opt_state['mb'] = [np.zeros_like(self.biases[i])  for i in range(L)]
			self._opt_state['vb'] = [np.zeros_like(self.biases[i])  for i in range(L)]

	# ---------- Activations and derivatives ----------
	def _act(self, z: np.ndarray, name: ActivationName) -> np.ndarray:
		if name == 'sigmoid':
			return 1.0 / (1.0 + np.exp(-z))
		elif name == 'relu':
			return np.maximum(0.0, z)
		elif name == 'leaky_relu':
			return np.where(z > 0.0, z, self.leaky_slope * z)
		elif name == 'linear':
			return z
		elif name == 'softmax':
			z_shift = z - np.max(z, axis=1, keepdims=True)
			exp_z = np.exp(z_shift)
			return exp_z / np.sum(exp_z, axis=1, keepdims=True)
		else:
			raise ValueError(f"Unknown activation: {name}")

	def _act_deriv(self, a: np.ndarray, z: np.ndarray, name: ActivationName) -> np.ndarray:
		if name == 'sigmoid':
			return a * (1.0 - a)
		elif name == 'relu':
			return (z > 0.0).astype(a.dtype)
		elif name == 'leaky_relu':
			out = np.ones_like(z)
			out[z < 0.0] = self.leaky_slope
			return out
		elif name == 'linear':
			return np.ones_like(z)
		elif name == 'softmax':
			raise RuntimeError("Softmax derivative handled via softmax loss simplification; don't call _act_deriv for softmax.")
		else:
			raise ValueError(f"Unknown activation: {name}")

	# ---------- Forward / Loss ----------
	def forward(self, X: np.ndarray) -> np.ndarray:
		self.zs = []
		self.as_ = [X]
		a = X
		for W, b, act_name in zip(self.weights, self.biases, self.activations_conf):
			z = a @ W + b
			a = self._act(z, act_name)
			self.zs.append(z)
			self.as_.append(a)
		return a

	def _regularization_loss(self) -> float:
		if self.reg is None or self.reg_lambda <= 0.0:
			return 0.0
		if self.reg == 'l2':
			return 0.5 * self.reg_lambda * sum(np.sum(W * W) for W in self.weights)
		elif self.reg == 'l1':
			return self.reg_lambda * sum(np.sum(np.abs(W)) for W in self.weights)
		else:
			raise ValueError(f"Unknown regularization: {self.reg}")

	def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		m = y_true.shape[0]
		eps = 1e-12
		if self.loss_name == 'mse':
			data_loss = np.mean((y_pred - y_true) ** 2)
		elif self.loss_name == 'softmax':
			y_pred_c = np.clip(y_pred, eps, 1 - eps)
			data_loss = -np.mean(np.sum(y_true * np.log(y_pred_c), axis=1))
		else:
			raise ValueError(f"Unknown loss: {self.loss_name}")
		reg_loss = self._regularization_loss()
		reg_loss = reg_loss / m if reg_loss > 0 else 0.0
		return float(data_loss + reg_loss)

	# ---------- Backward / Update ----------
	def _regularization_grad(self, W: np.ndarray) -> np.ndarray:
		if self.reg is None or self.reg_lambda <= 0.0:
			return np.zeros_like(W)
		if self.reg == 'l2':
			return self.reg_lambda * W
		elif self.reg == 'l1':
			return self.reg_lambda * np.sign(W)
		else:
			raise ValueError(f"Unknown regularization: {self.reg}")

	def _update_params(self, i: int, dW: np.ndarray, db: np.ndarray) -> None:
		if self.optimizer == 'gd':
			self.weights[i] -= self.learning_rate * dW
			self.biases[i]  -= self.learning_rate * db
		elif self.optimizer == 'rmsprop':
			sW = self._opt_state['sW'][i]
			sb = self._opt_state['sb'][i]
			sW = self.rho * sW + (1.0 - self.rho) * (dW * dW)
			sb = self.rho * sb + (1.0 - self.rho) * (db * db)
			self._opt_state['sW'][i] = sW
			self._opt_state['sb'][i] = sb
			self.weights[i] -= self.learning_rate * dW / (np.sqrt(sW) + self.eps)
			self.biases[i]  -= self.learning_rate * db / (np.sqrt(sb) + self.eps)
		elif self.optimizer == 'adam':
			mW = self._opt_state['mW'][i]; vW = self._opt_state['vW'][i]
			mb = self._opt_state['mb'][i]; vb = self._opt_state['vb'][i]
			mW = self.beta1 * mW + (1.0 - self.beta1) * dW
			vW = self.beta2 * vW + (1.0 - self.beta2) * (dW * dW)
			mb = self.beta1 * mb + (1.0 - self.beta1) * db
			vb = self.beta2 * vb + (1.0 - self.beta2) * (db * db)
			mW_hat = mW / (1.0 - self.beta1 ** self.t)
			vW_hat = vW / (1.0 - self.beta2 ** self.t)
			mb_hat = mb / (1.0 - self.beta1 ** self.t)
			vb_hat = vb / (1.0 - self.beta2 ** self.t)
			self.weights[i] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.eps)
			self.biases[i]  -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.eps)
			self._opt_state['mW'][i] = mW; self._opt_state['vW'][i] = vW
			self._opt_state['mb'][i] = mb; self._opt_state['vb'][i] = vb
		else:
			raise ValueError(f"Unknown optimizer: {self.optimizer}")

	def backward(self, y_true: np.ndarray) -> None:
		assert len(self.as_) >= 2, "Call forward(X) before backward()."
		m = y_true.shape[0]
		L = len(self.weights)
		a_L = self.as_[-1]
		act_L = self.activations_conf[-1]
		if self.loss_name == 'mse':
			if act_L == 'softmax':
				raise ValueError("MSE with softmax output is not supported; use softmax loss.")
			dz_L = (a_L - y_true) * self._act_deriv(a_L, self.zs[-1], act_L)
		elif self.loss_name == 'softmax':
			if act_L != 'softmax':
				raise ValueError("Softmax loss expects final activation 'softmax' and one-hot targets.")
			dz_L = (a_L - y_true)
		else:
			raise ValueError(f"Unknown loss: {self.loss_name}")
		deltas: List[np.ndarray] = [None] * L
		deltas[-1] = dz_L
		for i in range(L - 2, -1, -1):
			W_next = self.weights[i + 1]
			a_i1 = self.as_[i + 1]
			z_i = self.zs[i]
			act_i = self.activations_conf[i]
			da_i = deltas[i + 1] @ W_next.T
			dz_i = da_i * self._act_deriv(a_i1, z_i, act_i)
			deltas[i] = dz_i
		self.t += 1
		for i in range(L):
			a_prev = self.as_[i]
			d = deltas[i]
			dW = (a_prev.T @ d) / m + self._regularization_grad(self.weights[i]) / m
			db = np.sum(d, axis=0, keepdims=True) / m
			self._update_params(i, dW, db)

	# ---------- Training / Evaluation ----------
	def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
		y_pred = self.forward(X)
		return self._compute_loss(y, y_pred)

	def _mse_metric(self, y_true: np.ndarray, y_cont: np.ndarray) -> Optional[float]:
		yt = y_true
		yp = y_cont
		if yt.ndim == 1 and yp.ndim == 2 and yp.shape[1] == 1:
			yt = yt.reshape(-1, 1)
		if yt.shape == yp.shape:
			return float(np.mean((yp - yt) ** 2))
		return None

	def _accuracy_metric(self, y_true: np.ndarray, y_pred_disc: np.ndarray) -> Optional[float]:
		if self.loss_name != 'softmax':
			return None
		yt = y_true
		if isinstance(yt, np.ndarray) and yt.ndim == 2 and yt.shape[1] > 1:
			yt = np.argmax(yt, axis=1)
		yt = yt.ravel()
		yp = y_pred_disc.ravel()
		if yt.shape[0] != yp.shape[0]:
			return None
		return float(np.mean(yt == yp))

	def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
		y_cont = self.forward(X)
		loss = self._compute_loss(y, y_cont)
		if self.loss_name == 'softmax' and self.activations_conf[-1] == 'softmax':
			y_disc = np.argmax(y_cont, axis=1)
		else:
			y_disc = None
		mse = self._mse_metric(y, y_cont)
		acc = self._accuracy_metric(y, y_disc) if y_disc is not None else None
		return {"loss": float(loss), "mse": mse, "accuracy": acc}

	def train(
		self,
		X: np.ndarray,
		y: np.ndarray,
		epochs: int = 1000,
		batch_size: Optional[int] = None,
		verbose: bool = False,
		print_every: int = 100,
		shuffle: bool = True,
		log_metrics: bool = True,
	) -> Dict[str, List[Any]]:
		"""Train the network and optionally return history dict."""
		n = X.shape[0]
		if batch_size is None or batch_size <= 0 or batch_size >= n:
			batch_size = n
		history = {"epoch": [], "loss": [], "mse": [], "accuracy": []}
		for epoch in range(1, epochs + 1):
			idx = np.arange(n)
			if shuffle and batch_size < n:
				self.rng.shuffle(idx)
			for start in range(0, n, batch_size):
				end = min(start + batch_size, n)
				batch_idx = idx[start:end]
				Xb, yb = X[batch_idx], y[batch_idx]
				self.forward(Xb)
				self.backward(yb)
			if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
				if log_metrics:
					metrics = self.evaluate(X, y)
					history["epoch"].append(epoch)
					history["loss"].append(metrics["loss"])
					history["mse"].append(metrics["mse"])
					history["accuracy"].append(metrics["accuracy"])
					msg = f"Epoch {epoch:5d} | loss={metrics['loss']:.6f}"
					if metrics["mse"] is not None:
						msg += f" | mse={metrics['mse']:.6f}"
					if metrics["accuracy"] is not None:
						msg += f" | acc={metrics['accuracy']:.4f}"
					print(msg)
				else:
					loss_val = self.compute_loss(X, y)
					print(f"Epoch {epoch:5d} | loss={loss_val:.6f}")
		return history

	# ---------- Predictions ----------
	def predict(self, X: np.ndarray) -> np.ndarray:
		y_pred = self.forward(X)
		act_L = self.activations_conf[-1]
		if self.loss_name == 'softmax' and act_L == 'softmax':
			return np.argmax(y_pred, axis=1)
		else:
			return y_pred

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		return self.forward(X)


# Extend exports
__all__.extend([
	"FFNN",
	"one_hot",
])


# =============================================
# PyTorch MLP (PTMLP) for parity testing with FFNN
# =============================================
try:
	import torch
	import torch.nn as nn
except ImportError:  # torch optional; PTMLP only available if installed
	torch = None
	nn = None

if torch is not None:
	class PTMLP(nn.Module):
		"""Two-hidden-layer MLP matching FFNN initialization & activations.

		Architecture (default): [in_features, h1, h2, out_features]
		Hidden activation selectable: sigmoid / relu / leaky_relu
		Output: Linear for regression (MSE)

		Initialization (course-friendly fan-in scaling):
		- For ReLU/LeakyReLU: std = sqrt(2 / fan_in)
		- Otherwise (sigmoid/linear): std = 1 / sqrt(fan_in)
		Biases initialized to zero.
		"""
		def __init__(self, in_features=1, h1=100, h2=100, out_features=1, activation: str = 'sigmoid'):
			super().__init__()
			self.fc1 = nn.Linear(in_features, h1, bias=True)
			self.fc2 = nn.Linear(h1, h2, bias=True)
			self.fc3 = nn.Linear(h2, out_features, bias=True)
			if activation == 'sigmoid':
				self.act = nn.Sigmoid()
			elif activation == 'relu':
				self.act = nn.ReLU()
			elif activation == 'leaky_relu':
				self.act = nn.LeakyReLU(negative_slope=0.01)
			else:
				raise ValueError(f"Unsupported activation for PTMLP: {activation}")
			self._init_fanin(activation)

		def _init_fanin(self, activation: str):
			for m in self.modules():
				if isinstance(m, nn.Linear):
					fan_in = m.in_features
					if activation in ('relu', 'leaky_relu'):
						std = (2.0 / max(1, fan_in)) ** 0.5
					else:
						std = 1.0 / (max(1, fan_in)) ** 0.5
					nn.init.normal_(m.weight, mean=0.0, std=std)
					nn.init.constant_(m.bias, 0.0)

		def forward(self, x):
			x = self.act(self.fc1(x))
			x = self.act(self.fc2(x))
			return self.fc3(x)  # linear output layer

	__all__.append("PTMLP")

