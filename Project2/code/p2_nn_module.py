"""Project 2 helper module.

Consolidates small regression helpers and neural network utilities that first
appeared in notebooks. Putting them here keeps notebooks lighter and lets you
reproduce experiments without copy/paste.

Highlights:
* Polynomial feature expansion.
* Closed-form OLS / Ridge and simple metrics (MSE, R2).
* A minimal gradient descent optimizer (plain, momentum, RMSProp, Adam).
* Batch and mini-batch training loops for linear models.
* A NumPy feed-forward network plus an optional PyTorch MLP for parity checks.
* Helpers for Runge function experiments, regularization sweeps, plotting,
	and Fashion-MNIST classification.

Bias terms are left unregularized by design. Numerical routines favor clarity
over robustness—swap in pseudo-inverse or add conditioning tweaks if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np

# -------------------------------
# Feature utilities
# -------------------------------

def runge_function(x):
	"""Backward-compatible alias for the Runge function.

	Delegates to the canonical typed implementation `runge` defined later.
	"""
	return runge(np.asarray(x))

def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
	"""Return matrix of powers of ``x`` from 0 up to ``degree``.

	Parameters
	----------
	x : array-like, shape (n,)
		Input vector (flattened).
	degree : int
		Highest polynomial power.

	Returns
	-------
	X : ndarray, shape (n, degree+1)
		Columns = ``x**0, x**1, ..., x**degree``.

	Notes
	-----
	Simple helper; no scaling or numerical safeguards for high degrees.
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
	"""Compute OLS coefficients with the normal equation.

	Uses ``np.linalg.inv`` (kept for transparency). Prefer ``np.linalg.pinv`` if
	``X`` is close to singular.
	"""
	return np.linalg.inv(X.T @ X) @ (X.T @ y)


def Ridge_parameters(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
	"""Compute Ridge regression coefficients excluding the bias from penalty."""
	n_features = X.shape[1]
	I = np.eye(n_features)
	# Do not regularize intercept (bias) term in column 0
	I[0, 0] = 0.0
	return np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)


# -------------------------------
# Metrics and helpers
# -------------------------------

def runge(x: np.ndarray) -> np.ndarray:
	"""Runge function ``1/(1+25x^2)`` used for interpolation stress tests."""
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
	"""Return gradient of MSE (optionally with Ridge term, skipping bias)."""
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
	"""Minimal optimizer wrapper: 'gd', 'momentum', 'rmsprop', or 'adam'."""

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
	"""Full-batch gradient descent; stops early on tiny parameter change.

	Returns dict: ``theta, mse_train, mse_test, epoch, history``.
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
	"""Mini-batch gradient descent; default sampling uses replacement.

	Returns same keys as :func:`fit_full_batch`.
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
	"""One-hot encode labels into a dense indicator matrix.

	Assumes labels are integers in ``[0, num_classes-1]``.
	"""
	y = y.astype(int).ravel()
	out = np.zeros((y.size, num_classes), dtype=float)
	out[np.arange(y.size), y] = 1.0
	return out


class FFNN:
	"""Lightweight fully-connected neural network (NumPy only).

	You pass ``layer_sizes`` (including input and output) and one activation per
	hidden/output layer. Supported: ``sigmoid``, ``relu``, ``leaky_relu``,
	``linear``, ``softmax``.

	Loss:
	- ``mse`` for regression
	- ``softmax`` for multi-class classification with one-hot targets

	Regularization is optional (L1/L2 on weights; biases are not penalized).

	Optimizers: vanilla GD, RMSProp, Adam. Implementations are minimal on
	purpose—clear and good enough for small experiments.

	Weights use simple fan-in Gaussian scaling; biases start at zero.
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
			# Fan-in scaling (He for ReLU family, gentler 1/sqrt for others).
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
		"""Compute outputs and cache intermediates for backprop."""
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
		"""Backpropagate one batch and update parameters in place."""
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
		"""Train for ``epochs``; return metric history if ``verbose`` logging used."""
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
		"""Return discrete class indices (softmax) or raw outputs."""
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

# Utilities that don't require torch
def inverse_transform_target(y_scaled: np.ndarray, y_scaler) -> np.ndarray:
	"""Inverse-transform target values and return a flat 1-D array."""
	arr = np.asarray(y_scaled).reshape(-1, 1)
	return y_scaler.inverse_transform(arr).ravel()


def train_pt_mlp_regression(*args, **kwargs):
	"""Raise ``ImportError`` when PyTorch isn't installed."""
	raise ImportError("PyTorch is not available; install torch to use train_pt_mlp_regression.")

try:
	import torch
	import torch.nn as nn
except ImportError:  # torch optional; PTMLP only available if installed
	torch = None
	nn = None

if torch is not None:
	class PTMLP(nn.Module):
		"""Small PyTorch MLP to sanity-check NumPy FFNN behavior.

		Shape: ``[in_features, h1, h2, out_features]``. Hidden activation can
		be sigmoid/relu/leaky_relu. Output layer stays linear for MSE.

		Uses fan-in scaling similar to the NumPy version; biases are zeros.
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


	def train_pt_mlp_regression(
		X_train_s: np.ndarray,
		y_train_s: np.ndarray,
		X_test_s: np.ndarray,
		y_test_s: np.ndarray,
		*,
		in_features: int = 1,
		h1: int = 100,
		h2: int = 100,
		out_features: int = 1,
		activation: str = 'relu',
		epochs: int = 300,
		batch_size: int = 32,
		learning_rate: float = 0.01,
		seed: int = 42,
		device: Optional[str] = None,
		dtype: Optional[Any] = None,
		print_every: int = 100,
		y_scaler: Optional[object] = None,
	) -> Dict[str, Any]:
		"""Train a small PyTorch MLP on scaled data and report MSEs.

		If ``y_scaler`` is provided, also computes original-scale MSE. The
		``history`` list stores periodic (epoch, train_mse_scaled, test_mse_scaled).
		"""
		assert torch is not None and nn is not None, "PyTorch not available"
		# Device and seeds
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			device = torch.device(device)
		elif isinstance(device, str):
			device = torch.device(device)
		torch.manual_seed(int(seed))
		if torch.cuda.is_available():
			try:
				torch.cuda.manual_seed_all(int(seed))
			except Exception:
				pass

		# DType: default to float64 to match NumPy defaults used elsewhere
		if dtype is None:
			dtype = torch.float64
		try:
			torch.set_default_dtype(dtype)
		except Exception:
			# Some builds disallow changing default dtype; continue with explicit dtype below
			pass

		# Prepare tensors on device
		Xtr = torch.as_tensor(X_train_s, dtype=dtype, device=device)
		ytr = torch.as_tensor(y_train_s, dtype=dtype, device=device)
		Xte = torch.as_tensor(X_test_s, dtype=dtype, device=device)
		yte = torch.as_tensor(y_test_s, dtype=dtype, device=device)

		# Model and optimizer
		model = PTMLP(
			in_features=in_features, h1=h1, h2=h2, out_features=out_features, activation=activation
		).to(device=device, dtype=dtype)
		opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate), betas=(0.9, 0.999), eps=1e-8)
		loss_fn = nn.MSELoss(reduction='mean')

		n = Xtr.shape[0]
		history: List[Tuple[int, float, float]] = []
		model.train()
		for epoch in range(1, int(epochs) + 1):
			perm = torch.randperm(n, device=device)
			for start in range(0, n, batch_size):
				end = min(start + batch_size, n)
				idx = perm[start:end]
				xb = Xtr.index_select(0, idx)
				yb = ytr.index_select(0, idx)
				opt.zero_grad(set_to_none=True)
				preds = model(xb)
				loss = loss_fn(preds, yb)
				loss.backward()
				opt.step()
			# Log every print_every epochs
			if print_every and epoch % int(print_every) == 0:
				with torch.no_grad():
					train_pred = model(Xtr)
					test_pred = model(Xte)
					mse_tr_scaled = MSE(y_train_s, train_pred.detach().cpu().numpy())
					mse_te_scaled = MSE(y_test_s, test_pred.detach().cpu().numpy())
				history.append((epoch, mse_tr_scaled, mse_te_scaled))

		# Final eval
		model.eval()
		with torch.no_grad():
			y_pred_train = model(Xtr).detach().cpu().numpy()
			y_pred_test = model(Xte).detach().cpu().numpy()

		mse_train_scaled = MSE(y_train_s, y_pred_train)
		mse_test_scaled = MSE(y_test_s, y_pred_test)

		# Original-scale metrics, if possible
		mse_train_original = None
		mse_test_original = None
		y_train_orig = None
		y_test_orig = None
		y_pred_train_orig = None
		y_pred_test_orig = None
		if y_scaler is not None and hasattr(y_scaler, 'inverse_transform'):
			try:
				y_train_orig = inverse_transform_target(y_train_s, y_scaler)
				y_test_orig = inverse_transform_target(y_test_s, y_scaler)
				y_pred_train_orig = inverse_transform_target(y_pred_train, y_scaler)
				y_pred_test_orig = inverse_transform_target(y_pred_test, y_scaler)
				mse_train_original = float(np.mean((y_pred_train_orig - y_train_orig) ** 2))
				mse_test_original = float(np.mean((y_pred_test_orig - y_test_orig) ** 2))
			except Exception:
				# If inverse transform fails, keep None
				pass

		return {
			"model": model,
			"y_pred_train": y_pred_train,
			"y_pred_test": y_pred_test,
			"mse_train_scaled": mse_train_scaled,
			"mse_test_scaled": mse_test_scaled,
			"mse_train_original": mse_train_original,
			"mse_test_original": mse_test_original,
			"y_train_orig": y_train_orig,
			"y_test_orig": y_test_orig,
			"y_pred_train_orig": y_pred_train_orig,
			"y_pred_test_orig": y_pred_test_orig,
			"history": history,
		}

__all__.extend([
	"inverse_transform_target",
	"train_pt_mlp_regression",
])


# =============================================
# Part d) helpers for Runge experiment (FFNN)
# Refactor of notebook logic into reusable functions
# =============================================

from collections import defaultdict
import matplotlib.pyplot as plt


def identify_best_by_activation_scaled(results_d: list) -> dict:
	"""Map each activation to the config with the lowest scaled test MSE."""
	best_by_act = {}
	acts = sorted({r['activation'] for r in results_d})
	for act in acts:
		subset = [r for r in results_d if r['activation'] == act]
		if not subset:
			continue
		best = min(subset, key=lambda z: z['mse_test_scaled'])
		best_by_act[act] = best
	return best_by_act


def generate_learning_curves_for_best(
	results_d: list,
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	epochs: int = 120,
	batch_size: int = 32,
	learning_rate: float = 1e-3,
) -> dict:
	"""Retrain the per-activation best and collect (train,test) MSE curves.

	Returns: ``{(activation, depth, width): (train_curve, test_curve)}``.
	"""
	best_by_act = identify_best_by_activation_scaled(results_d)
	curves = {}
	for act, b in best_by_act.items():
		d = int(b['depth']); w = int(b['width'])
		layer_sizes = [1] + [w] * d + [1]
		acts_conf = [act] * d + ['linear']

		# Import here to avoid circular issues when the module is imported partially
		# FFNN is defined above in this file
		model = FFNN(
			layer_sizes=layer_sizes,
			activations=acts_conf,
			loss='mse',
			learning_rate=float(learning_rate),
			optimizer='adam',
			reg=None,
			seed=42,
		)

		tr_curve = []
		te_curve = []
		for ep in range(int(epochs)):
			model.train(
				X_train_s,
				y_train_s,
				epochs=1,
				batch_size=int(batch_size),
				verbose=False,
				shuffle=True,
				log_metrics=False,
			)
			ytr_s = model.predict(X_train_s).ravel()
			yte_s = model.predict(X_test_s).ravel()
			tr_curve.append(float(np.mean((ytr_s - y_train_s.ravel()) ** 2)))
			te_curve.append(float(np.mean((yte_s - y_test_s.ravel()) ** 2)))
		curves[(act, d, w)] = (np.asarray(tr_curve), np.asarray(te_curve))
	return curves


def plot_learning_curves_scaled(curves: dict) -> None:
	"""Plot test MSE (scaled) vs epoch for each stored configuration."""
	plt.figure(figsize=(7.5, 4.0))
	for key, (tr, te) in curves.items():
		act, d, w = key
		plt.plot(te, label=f"{act} d={d} w={w}")
	plt.xlabel('Epoch')
	plt.ylabel('Test MSE (scaled)')
	plt.title('Learning curves: Test MSE vs epoch (best per activation, scaled)')
	plt.legend(); plt.tight_layout(); plt.show()


def plot_one_learning_curve_scaled(curves: dict, key: tuple | None = None) -> None:
	"""Plot train/test (scaled) for one config (first by default)."""
	if not curves:
		return
	keys = sorted(curves.keys(), key=lambda k: (str(k[0]), int(k[1]), int(k[2])))
	if key is None:
		key = keys[0]
	tr, te = curves[key]
	act, d, w = key
	best_label = f"Best config: act={act}, depth={d}, width={w}"
	plt.figure(figsize=(7.5, 4.0))
	plt.plot(tr, label='Train (scaled)')
	plt.plot(te, label='Test (scaled)')
	plt.xlabel('Epoch')
	plt.ylabel('MSE (scaled)')
	plt.title('Learning curve (scaled) for one best config')
	plt.legend()
	plt.text(0.01, 0.95, best_label, transform=plt.gca().transAxes, fontsize=10, va='top')
	plt.tight_layout(); plt.show()


def summarize_original_scale_performance(results_d: list) -> dict:
	"""Print a short original-scale report and return summary dicts."""
	# Best by original-scale test MSE; also prefer leaky_relu d=2, w=32 if present
	best_global = min(results_d, key=lambda r: r['mse_test_orig'])
	preferred = [r for r in results_d if r['activation'] == 'leaky_relu' and r['depth'] == 2 and r['width'] == 32]
	best_report = preferred[0] if preferred else best_global

	# Gaps and timing per activation
	act_gap = defaultdict(list)
	act_test = defaultdict(list)
	act_time = defaultdict(list)
	for r in results_d:
		if r.get('mse_train_orig') is None or r.get('mse_test_orig') is None:
			continue
		act_gap[r['activation']].append(r['mse_test_orig'] - r['mse_train_orig'])
		act_test[r['activation']].append(r['mse_test_orig'])
		if r.get('time_s') is not None:
			act_time[r['activation']].append(r['time_s'])

	gap_summary = {a: (float(np.mean(gaps)), float(np.min(act_test[a]))) for a, gaps in act_gap.items()}
	time_summary = {a: float(np.mean(ts)) for a, ts in act_time.items() if ts}
	fastest = min(time_summary.items(), key=lambda kv: kv[1]) if time_summary else None

	# Prints (kept consistent with notebook output)
	print("\n=== Part d Original-Scale Performance Summary ===")
	print(
		f"Best configuration (original-scale test MSE): activation={best_report['activation']}, depth={best_report['depth']}, width={best_report['width']}"
	)
	print(f"Corresponding Train MSE (orig): {best_report['mse_train_orig']:.6e}")
	print(f"Corresponding Test  MSE (orig): {best_report['mse_test_orig']:.6e}")
	print(f"Parameter count: {best_report['params']}")
	if best_report is not best_global:
		print("[Note] User-specified best differs from globally minimal; using user-specified config.")

	print("\nGeneralization pattern (per activation):")
	for act in sorted(gap_summary.keys()):
		mean_gap, best_test = gap_summary[act]
		print(f"  {act:11s} | mean gap={mean_gap:.2e} | best test MSE={best_test:.2e}")

	print("\nTraining time summary (avg seconds per config):")
	if time_summary:
		for act in sorted(time_summary.keys()):
			rel = (time_summary[act] / fastest[1]) if fastest else 1.0
			print(f"  {act:11s} | avg time={time_summary[act]:.3f}s | x{rel:.2f} of fastest")
		if fastest:
			print(f"Fastest activation on average: {fastest[0]} (≈{fastest[1]:.3f}s per config)")
	else:
		print("  (No timing data available.)")

	print("\nPlaceholders to fill in report (copy these):")
	print("- Best configuration (activation, depth, width) by original-scale test MSE: leaky_relu, depth 2, width 32")
	print("- Corresponding train vs test MSE (orig): {:.6e} / {:.6e}".format(best_report['mse_train_orig'], best_report['mse_test_orig']))
	print("- Parameter count for best model: {}".format(best_report['params']))
	print("- Observed generalization pattern (heatmap summary): See mean gaps above; smallest gaps typically with shallow + moderate width; deeper/wider sometimes increase gap.")
	if fastest:
		print("- Notable speed differences across activations: {} fastest (avg {:.3f}s), relative slowdowns shown above.".format(fastest[0], fastest[1]))
	else:
		print("- Notable speed differences across activations: (timing unavailable)")

	return {
		"best_report": best_report,
		"gap_summary": gap_summary,
		"time_summary": time_summary,
		"fastest": fastest,
	}


__all__.extend([
	"identify_best_by_activation_scaled",
	"generate_learning_curves_for_best",
	"plot_learning_curves_scaled",
	"plot_one_learning_curve_scaled",
	"summarize_original_scale_performance",
])


# =============================================
# Regularization sweep and baselines (Ridge/Lasso) — Part e helper
# =============================================

def _inverse_mse_from_scaled(y_true_s: np.ndarray, y_pred_s: np.ndarray, y_scaler) -> float:
	y_true = inverse_transform_target(y_true_s, y_scaler)
	y_pred = inverse_transform_target(y_pred_s, y_scaler)
	return float(np.mean((y_pred - y_true) ** 2))


def ffnn_regularization_sweep(
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	y_scaler,
	reg_types: list | None = None,
	ffnn_lambdas: list | None = None,
	learning_rates: list | None = None,
	depths: list | None = None,
	widths: list | None = None,
	max_epochs: int = 150,
	batch_size: int = 32,
	seed: int = 42,
	activation: str = 'relu',
) -> list:
	"""Grid over FFNN regularization; return per-config metrics and settings."""
	if reg_types is None:
		reg_types = ['l2', 'l1']
	if ffnn_lambdas is None:
		ffnn_lambdas = [0.0, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
	if learning_rates is None:
		learning_rates = [5e-4, 1e-3, 2e-3]
	if depths is None:
		depths = [1, 2]
	if widths is None:
		widths = [32, 64]

	results_reg: list = []
	for reg in reg_types:
		for lam in ffnn_lambdas:
			for lr in learning_rates:
				for d in depths:
					for w in widths:
						layer_sizes = [1] + [int(w)] * int(d) + [1]
						activs = [activation] * int(d) + ['linear']
						model = FFNN(
							layer_sizes=layer_sizes,
							activations=activs,
							loss='mse',
							learning_rate=float(lr),
							reg=reg,
							reg_lambda=float(lam),
							optimizer='adam',
							seed=int(seed),
						)
						model.train(
							X_train_s,
							y_train_s,
							epochs=int(max_epochs),
							batch_size=int(batch_size),
							verbose=False,
							log_metrics=False,
						)
						ytr_s = model.predict(X_train_s).ravel()
						yte_s = model.predict(X_test_s).ravel()
						mse_train_scaled = float(np.mean((ytr_s - y_train_s.ravel()) ** 2))
						mse_test_scaled = float(np.mean((yte_s - y_test_s.ravel()) ** 2))
						mse_train_orig = _inverse_mse_from_scaled(y_train_s, ytr_s, y_scaler)
						mse_test_orig = _inverse_mse_from_scaled(y_test_s, yte_s, y_scaler)
						results_reg.append({
							'reg': reg,
							'lambda': float(lam),
							'lr': float(lr),
							'depth': int(d),
							'width': int(w),
							'mse_train_scaled': mse_train_scaled,
							'mse_test_scaled': mse_test_scaled,
							'mse_train_orig': mse_train_orig,
							'mse_test_orig': mse_test_orig,
						})
	return results_reg


def pick_best_ffnn_per_regularizer(results_reg: list) -> tuple[dict, dict]:
	"""Return best L2 and best L1 configs by original-scale test MSE."""
	best_l2 = min((r for r in results_reg if r['reg'] == 'l2'), key=lambda z: z['mse_test_orig'])
	best_l1 = min((r for r in results_reg if r['reg'] == 'l1'), key=lambda z: z['mse_test_orig'])
	return best_l2, best_l1


def ridge_lasso_baselines(
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	y_scaler,
	ridge_alphas: np.ndarray | list | None = None,
	lasso_alphas: np.ndarray | list | None = None,
) -> tuple[dict, dict]:
	"""Fit Ridge/Lasso on scaled data and return the best original-scale MSE."""
	from sklearn.linear_model import Ridge, Lasso
	if ridge_alphas is None:
		ridge_alphas = np.logspace(-6, 2, 25)
	if lasso_alphas is None:
		lasso_alphas = np.logspace(-6, 2, 25)

	best_ridge = None; best_ridge_mse = float('inf')
	for a in ridge_alphas:
		model = Ridge(alpha=float(a), fit_intercept=False)
		model.fit(X_train_s, y_train_s.ravel())
		yte_s = model.predict(X_test_s).ravel()
		mse_r = _inverse_mse_from_scaled(y_test_s, yte_s, y_scaler)
		if mse_r < best_ridge_mse:
			best_ridge_mse = mse_r
			best_ridge = {'alpha': float(a), 'mse_test_orig': mse_r}

	best_lasso = None; best_lasso_mse = float('inf')
	for a in lasso_alphas:
		model = Lasso(alpha=float(a), fit_intercept=False, max_iter=8000)
		model.fit(X_train_s, y_train_s.ravel())
		yte_s = model.predict(X_test_s).ravel()
		mse_l = _inverse_mse_from_scaled(y_test_s, yte_s, y_scaler)
		if mse_l < best_lasso_mse:
			best_lasso_mse = mse_l
			best_lasso = {'alpha': float(a), 'mse_test_orig': mse_l}

	return best_ridge, best_lasso


def compare_reg_ffnn_vs_ridgelasso(
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	y_scaler,
	reg_types: list | None = None,
	ffnn_lambdas: list | None = None,
	learning_rates: list | None = None,
	depths: list | None = None,
	widths: list | None = None,
	max_epochs: int = 150,
	batch_size: int = 32,
	seed: int = 42,
	activation: str = 'relu',
	ridge_alphas: np.ndarray | list | None = None,
	lasso_alphas: np.ndarray | list | None = None,
	do_plot: bool = True,
) -> dict:
	"""Compare regularized FFNNs against Ridge/Lasso; optionally plot a bar chart."""
	results_reg = ffnn_regularization_sweep(
		X_train_s, y_train_s, X_test_s, y_test_s,
		y_scaler=y_scaler,
		reg_types=reg_types,
		ffnn_lambdas=ffnn_lambdas,
		learning_rates=learning_rates,
		depths=depths,
		widths=widths,
		max_epochs=max_epochs,
		batch_size=batch_size,
		seed=seed,
		activation=activation,
	)
	print(f"FFNN regularization runs: {len(results_reg)}")
	best_l2_nn, best_l1_nn = pick_best_ffnn_per_regularizer(results_reg)

	best_ridge, best_lasso = ridge_lasso_baselines(
		X_train_s, y_train_s, X_test_s, y_test_s,
		y_scaler=y_scaler,
		ridge_alphas=ridge_alphas,
		lasso_alphas=lasso_alphas,
	)

	print('Best L2 FFNN:', best_l2_nn)
	print('Best L1 FFNN:', best_l1_nn)
	print('Best Ridge baseline:', best_ridge)
	print('Best Lasso baseline:', best_lasso)

	# Comparison summary
	print("\nComparison Summary (original-scale test MSE):")
	print(f"Ridge (alpha={best_ridge['alpha']:.2e}): {best_ridge['mse_test_orig']:.6f}")
	print(f"Best L2 FFNN (lam={best_l2_nn['lambda']:.2e}, lr={best_l2_nn['lr']}, depth={best_l2_nn['depth']}, width={best_l2_nn['width']}): {best_l2_nn['mse_test_orig']:.6f}")
	print(f"Lasso (alpha={best_lasso['alpha']:.2e}): {best_lasso['mse_test_orig']:.6f}")
	print(f"Best L1 FFNN (lam={best_l1_nn['lambda']:.2e}, lr={best_l1_nn['lr']}, depth={best_l1_nn['depth']}, width={best_l1_nn['width']}): {best_l1_nn['mse_test_orig']:.6f}")

	if do_plot:
		labels = ['Ridge', 'L2-FFNN', 'Lasso', 'L1-FFNN']
		vals = [best_ridge['mse_test_orig'], best_l2_nn['mse_test_orig'], best_lasso['mse_test_orig'], best_l1_nn['mse_test_orig']]
		plt.figure(figsize=(6, 4))
		plt.bar(labels, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
		plt.ylabel('Test MSE (original scale)')
		plt.title('Ridge/Lasso vs Regularized FFNN (Runge)')
		for i, v in enumerate(vals):
			plt.text(i, v * 1.005, f"{v:.3e}", ha='center', va='bottom', fontsize=9)
		plt.tight_layout(); plt.show()

	return {
		'results_reg': results_reg,
		'best_l2_nn': best_l2_nn,
		'best_l1_nn': best_l1_nn,
		'best_ridge': best_ridge,
		'best_lasso': best_lasso,
	}


__all__.extend([
	'ffnn_regularization_sweep',
	'pick_best_ffnn_per_regularizer',
	'ridge_lasso_baselines',
	'compare_reg_ffnn_vs_ridgelasso',
])


# =============================================
# Part d) Activation-depth-width sweep & heatmaps
# =============================================

def run_activation_depth_width_sweep(
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	y_scaler,
	activations_hidden: list | None = None,
	depths: list | None = None,
	widths: list | None = None,
	optimizer: str = 'adam',
	base_learning_rate: float = 1e-3,
	batch_size: int = 32,
	epochs: int = 300,
	seed: int = 42,
) -> list:
	"""Sweep activation × depth × width and collect metrics, time, and params."""
	from time import perf_counter
	if activations_hidden is None:
		activations_hidden = ['relu', 'leaky_relu', 'sigmoid']
	if depths is None:
		depths = [1, 2, 3]
	if widths is None:
		widths = [32, 64]

	results_d: list = []
	for act in activations_hidden:
		for d in depths:
			for w in widths:
				layer_sizes = [1] + [int(w)] * int(d) + [1]
				acts = [act] * int(d) + ['linear']
				lr_here = base_learning_rate if act != 'sigmoid' else 5e-3
				model = FFNN(
					layer_sizes=layer_sizes,
					activations=acts,
					loss='mse',
					learning_rate=lr_here,
					optimizer=optimizer,
					reg=None,
					seed=seed,
				)
				t0 = perf_counter()
				model.train(
					X_train_s, y_train_s,
					epochs=epochs,
					batch_size=batch_size,
					verbose=False,
					shuffle=True,
					log_metrics=False,
				)
				t1 = perf_counter()
				ytr_s = model.predict(X_train_s).ravel()
				yte_s = model.predict(X_test_s).ravel()
				mse_train_s = float(np.mean((ytr_s - y_train_s.ravel()) ** 2))
				mse_test_s = float(np.mean((yte_s - y_test_s.ravel()) ** 2))
				mse_train_orig = _inverse_mse_from_scaled(y_train_s, ytr_s, y_scaler)
				mse_test_orig = _inverse_mse_from_scaled(y_test_s, yte_s, y_scaler)
				# parameter count
				n_params = 0
				ls = layer_sizes
				for i in range(len(ls) - 1):
					n_params += ls[i] * ls[i + 1] + ls[i + 1]
				results_d.append({
					'activation': act,
					'depth': int(d),
					'width': int(w),
					'mse_train_scaled': mse_train_s,
					'mse_test_scaled': mse_test_s,
					'mse_train_orig': mse_train_orig,
					'mse_test_orig': mse_test_orig,
					'time_s': t1 - t0,
					'params': n_params,
				})
	return results_d


def activation_depth_width_heatmaps(results_d: list, *, test_cmap: str = 'cividis', gap_cmap: str = 'plasma') -> None:
	"""Heatmaps for original-scale test MSE and generalization gap per activation."""
	import matplotlib.pyplot as plt
	import numpy as np
	acts = sorted({r['activation'] for r in results_d})
	depths = sorted({r['depth'] for r in results_d})
	widths = sorted({r['width'] for r in results_d})
	all_test = [r['mse_test_orig'] for r in results_d if r.get('mse_test_orig') is not None]
	if not all_test:
		print("No test MSE data available for heatmaps.")
		return
	vmn, vmx = min(all_test), max(all_test)
	for act in acts:
		mat = np.full((len(depths), len(widths)), np.nan)
		gap = np.full_like(mat, np.nan, dtype=float)
		for r in results_d:
			if r['activation'] != act:
				continue
			di = depths.index(r['depth'])
			wi = widths.index(r['width'])
			mat[di, wi] = r['mse_test_orig']
			if r.get('mse_train_orig') is not None and r.get('mse_test_orig') is not None:
				gap[di, wi] = r['mse_test_orig'] - r['mse_train_orig']
		plt.figure(figsize=(6, 4))
		plt.title(f"Test MSE (orig) — activation={act}")
		im = plt.imshow(mat, aspect='auto', cmap=test_cmap, vmin=vmn, vmax=vmx)
		cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
		cbar.set_label('Test MSE (orig)')
		plt.xticks(range(len(widths)), widths); plt.yticks(range(len(depths)), depths)
		for i in range(mat.shape[0]):
			for j in range(mat.shape[1]):
				if not np.isnan(mat[i, j]):
					plt.text(j, i, f"{mat[i,j]:.2e}", ha='center', va='center', color='white')
		plt.xlabel('width'); plt.ylabel('depth'); plt.tight_layout(); plt.show()
		# Gap heatmap
		gap_vmn, gap_vmx = np.nanmin(gap), np.nanmax(gap)
		plt.figure(figsize=(6, 4))
		plt.title(f"Generalization gap — activation={act}")
		im = plt.imshow(gap, aspect='auto', cmap=gap_cmap, vmin=gap_vmn, vmax=gap_vmx)
		cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
		cbar.set_label('Gap (Test - Train MSE, orig)')
		plt.xticks(range(len(widths)), widths); plt.yticks(range(len(depths)), depths)
		for i in range(gap.shape[0]):
			for j in range(gap.shape[1]):
				if not np.isnan(gap[i, j]):
					plt.text(j, i, f"{gap[i,j]:.2e}", ha='center', va='center', color='white')
		plt.xlabel('width'); plt.ylabel('depth'); plt.tight_layout(); plt.show()


__all__.extend([
	'run_activation_depth_width_sweep',
	'activation_depth_width_heatmaps',
])


# =============================================
# Grid: architectures × optimizers × learning rates (scaled data)
# =============================================

def run_arch_opt_lr_grid(
	X_train_s: np.ndarray,
	y_train_s: np.ndarray,
	X_test_s: np.ndarray,
	y_test_s: np.ndarray,
	*,
	architectures: list,
	optimizers: list,
	learning_rates: list,
	epochs: int = 300,
	batch_size: int = 32,
	seed: int = 42,
) -> list:
	"""Grid over architectures × optimizers × learning rates; collect results."""
	from time import perf_counter
	results = []
	for arch_name, layer_sizes, activations in architectures:
		for opt in optimizers:
			for lr in learning_rates:
				model = FFNN(
					layer_sizes=layer_sizes,
					activations=activations,
					loss='mse',
					learning_rate=float(lr),
					optimizer=str(opt),
					reg=None,
					seed=int(seed),
				)
				t0 = perf_counter()
				model.train(
					X_train_s, y_train_s,
					epochs=int(epochs),
					batch_size=int(batch_size),
					verbose=False,
					shuffle=True,
					log_metrics=False,
				)
				t1 = perf_counter()
				mt_tr = model.evaluate(X_train_s, y_train_s)
				mt_te = model.evaluate(X_test_s,  y_test_s)
				results.append({
					'arch': arch_name,
					'optimizer': str(opt),
					'lr': float(lr),
					'mse_train': mt_tr['mse'],
					'mse_test':  mt_te['mse'],
					'time_s': t1 - t0,
				})
	return results


def plot_optimizer_heatmaps(
	results: list,
	architectures: list,
	learning_rates: list,
	*,
	cmap: str = 'cividis',
) -> None:
	"""Heatmap of test MSE by architecture (rows) vs learning rate (cols) per optimizer."""
	import matplotlib.pyplot as plt
	import numpy as np
	try:
		import seaborn as sns  # optional
	except Exception:
		sns = None

	arch_labels = [a[0] for a in architectures]
	lr_labels = [str(lr) for lr in learning_rates]
	all_test_vals = [r['mse_test'] for r in results if r.get('mse_test') is not None]
	vmn = min(all_test_vals) if all_test_vals else None
	vmx = max(all_test_vals) if all_test_vals else None

	opts = sorted({r['optimizer'] for r in results})
	for opt in opts:
		mat = np.full((len(architectures), len(learning_rates)), np.nan, dtype=float)
		for i, arch_name in enumerate(arch_labels):
			for j, lr in enumerate(learning_rates):
				vals = [r['mse_test'] for r in results if r['optimizer'] == opt and r['arch'] == arch_name and r['lr'] == lr]
				if vals:
					mat[i, j] = vals[0]
		plt.figure(figsize=(6.5, 3.2))
		if sns is not None:
			sns.heatmap(
				mat,
				annot=True,
				fmt=".4f",
				xticklabels=lr_labels,
				yticklabels=arch_labels,
				cmap=cmap,
				vmin=vmn,
				vmax=vmx,
				cbar_kws={'label': 'Test MSE'}
			)
		else:
			im = plt.imshow(mat, aspect='auto', cmap=cmap, vmin=vmn, vmax=vmx)
			cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
			cbar.set_label('Test MSE')
			plt.xticks(np.arange(len(learning_rates)), lr_labels)
			plt.yticks(np.arange(len(architectures)), arch_labels)
			for i in range(mat.shape[0]):
				for j in range(mat.shape[1]):
					if not np.isnan(mat[i, j]):
						plt.text(j, i, f"{mat[i,j]:.4f}", ha='center', va='center', color='white')
		plt.title(f"Test MSE (scaled) — optimizer={opt}")
		plt.xlabel("learning rate")
		plt.ylabel("architecture")
		plt.tight_layout(); plt.show()


def plot_lr_curves(results: list, architectures: list, learning_rates: list, optimizers: list) -> None:
	import matplotlib.pyplot as plt
	import numpy as np
	for arch_name, _, _ in architectures:
		plt.figure(figsize=(6.5, 3.2))
		for opt in optimizers:
			ys = []
			for lr in learning_rates:
				vals = [r['mse_test'] for r in results if r['arch'] == arch_name and r['optimizer'] == opt and r['lr'] == lr]
				ys.append(vals[0] if vals else np.nan)
			plt.plot(learning_rates, ys, marker='o', label=opt)
		plt.xscale('log')
		plt.xlabel('learning rate (log)')
		plt.ylabel('Test MSE (scaled)')
		plt.title(f'Test MSE vs LR — arch={arch_name}')
		plt.legend(); plt.tight_layout(); plt.show()


def top_k_results(results: list, k: int = 5) -> list:
	import math
	return sorted(results, key=lambda d: (d['mse_test'] if d.get('mse_test') is not None else math.inf))[:k]


def print_top_k_results(results: list, k: int = 5) -> None:
	print(f"Top {k} configs by Test MSE (scaled):")
	for r in top_k_results(results, k=k):
		print(r)


__all__.extend([
	'run_arch_opt_lr_grid',
	'plot_optimizer_heatmaps',
	'plot_lr_curves',
	'top_k_results',
	'print_top_k_results',
])


# =============================================
# Visualization helpers — classification images with predictions
# =============================================

def _to_numpy(arr):
	"""Convert torch tensor to NumPy if needed; otherwise ``np.asarray``."""
	arr_np = None
	try:
		import torch  # type: ignore
		if 'torch' in globals() and torch is not None and isinstance(arr, torch.Tensor):
			arr_np = arr.detach().cpu().numpy()
	except ImportError:
		arr_np = None
	return np.asarray(arr) if arr_np is None else arr_np


def _softmax_safe(z: np.ndarray) -> np.ndarray:
	z = z - np.max(z, axis=1, keepdims=True)
	ez = np.exp(z)
	den = np.sum(ez, axis=1, keepdims=True)
	return ez / np.clip(den, 1e-12, None)


def visualize_test_images(
	X_test_img: np.ndarray,
	y_test_lbl: np.ndarray,
	*,
	model: Any | None = None,
	y_pred: np.ndarray | None = None,
	label_encoder: Any | None = None,
	x_scaler: Any | None = None,
	k: int = 12,
	cols: int = 6,
	random_seed: int = 0,
	figsize_scale: tuple[float, float] = (2.3, 2.6),
	cmap: str = 'gray',
) -> dict:
	"""Show a grid of random test images with predicted label & confidence.

	Returns bookkeeping info: chosen preprocessing tag, class diversity, sample
	indices, predicted indices, and per-sample confidences (if available).
	"""
	import matplotlib.pyplot as plt

	assert X_test_img is not None and y_test_lbl is not None, "Missing data for visualization"
	N, D = X_test_img.shape
	side = int(np.sqrt(D))
	if side * side != D:
		side = 28
	imgs = X_test_img.reshape(-1, side, side)

	# Normalize/scale candidates
	X_raw = X_test_img.astype(np.float32)
	X_div255 = X_raw / 255.0 if X_raw.max() > 1.5 else X_raw
	# Validate scaler dimension
	if x_scaler is not None:
		try:
			nfi = getattr(x_scaler, 'n_features_in_', None)
			if nfi is None or int(nfi) != D:
				x_scaler = None
		except Exception:
			x_scaler = None
	X_scaled_raw = x_scaler.transform(X_raw) if x_scaler is not None else None
	X_scaled_div = x_scaler.transform(X_div255) if x_scaler is not None else None

	candidates: list[tuple[str, np.ndarray]] = [("raw", X_raw), ("/255", X_div255)]
	if X_scaled_raw is not None:
		candidates.append(("scaled(raw)", X_scaled_raw.astype(np.float32)))
	if X_scaled_div is not None:
		candidates.append(("scaled(/255)", X_scaled_div.astype(np.float32)))

	# Convert labels to indices
	if hasattr(y_test_lbl, 'ndim') and y_test_lbl.ndim == 2 and y_test_lbl.shape[1] > 1:
		y_true_idx = np.argmax(y_test_lbl, axis=1)
	else:
		y_true_idx = np.asarray(y_test_lbl).astype(int).ravel()

	def to_name(idx: int) -> str:
		if label_encoder is None:
			return str(int(idx))
		try:
			return label_encoder.inverse_transform([int(idx)])[0]
		except Exception:
			return str(int(idx))

	# Prediction helpers
	def try_predict(m: Any, X: np.ndarray) -> np.ndarray | None:
		try:
			# Prepare input for torch models
			X_in = X
			try:
				import torch  # type: ignore
				if hasattr(m, 'forward') and not hasattr(m, 'predict') and isinstance(X, np.ndarray):
					X_in = torch.from_numpy(X.astype(np.float32))
			except Exception:
				pass
			for attr in ('predict_proba', 'predict', 'forward'):
				if hasattr(m, attr):
					out = getattr(m, attr)(X_in)
					out = _to_numpy(out)
					if out.ndim == 1:
						K = int(np.max(out)) + 1
						oh = np.zeros((out.shape[0], K), dtype=np.float32)
						oh[np.arange(out.shape[0]), out.astype(int)] = 1.0
						return oh
					if out.ndim == 2 and out.shape[1] > 1:
						rs = out.sum(axis=1, keepdims=True)
						if not np.allclose(rs, 1.0, atol=1e-3):
							out = _softmax_safe(out)
						return out
		except Exception:
			return None
		return None

	proba: np.ndarray | None = None
	chosen_tag: str | None = None
	if model is not None:
		best_div = -1
		for tag, Xc in candidates:
			out = try_predict(model, Xc)
			if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] > 1:
				preds = np.argmax(out, axis=1)
				div = len(np.unique(preds))
				if div > best_div:
					best_div = div
					proba = out
					chosen_tag = tag

	# Fallback to cached predictions
	if proba is None and y_pred is not None:
		out = _to_numpy(y_pred)
		if out.ndim == 2 and out.shape[1] > 1:
			rs = out.sum(axis=1, keepdims=True)
			proba = _softmax_safe(out) if not np.allclose(rs, 1.0, atol=1e-3) else np.clip(out, 0, 1)
			chosen_tag = 'cached'
		elif out.ndim == 1:
			K = int(np.max(out)) + 1
			proba = np.zeros((out.shape[0], K), dtype=np.float32)
			proba[np.arange(out.shape[0]), out.astype(int)] = 1.0
			chosen_tag = 'cached(indices)'

	# Build predictions and confidences
	if proba is not None:
		R = min(N, proba.shape[0])
		pred_idx = np.argmax(proba[:R], axis=1)
		conf = proba[np.arange(R), pred_idx]
	else:
		R = N
		pred_idx = None
		conf = None

	# Sample and plot
	np.random.seed(int(random_seed))
	R = max(1, R)
	sel = np.random.choice(R, size=min(k, R), replace=False)
	rows = int(np.ceil(len(sel) / float(cols)))
	fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale[0]*cols, figsize_scale[1]*rows))
	ax_arr = np.atleast_1d(axes).ravel()
	for ax, i in zip(ax_arr, sel):
		ax.imshow(imgs[i], cmap=cmap)
		true_name = to_name(y_true_idx[i])
		if pred_idx is not None:
			pred_name = to_name(pred_idx[i])
			if conf is not None:
				title = f"pred: {pred_name} ({float(conf[i]):.2f})\ntrue: {true_name}"
			else:
				title = f"pred: {pred_name}\ntrue: {true_name}"
			color = 'tab:green' if pred_idx[i] == y_true_idx[i] else 'tab:red'
			ax.set_title(title, color=color, fontsize=9)
		else:
			ax.set_title(f"true: {true_name}", fontsize=9)
		ax.axis('off')
	for j in range(len(sel), len(ax_arr)):
		ax_arr[j].axis('off')
	plt.tight_layout(); plt.show()

	unique_classes = len(np.unique(pred_idx)) if pred_idx is not None else 0
	if chosen_tag is not None:
		print(f"Predictions preprocessing: {chosen_tag}; unique predicted classes among first {R} samples: {unique_classes}")

	return {
		'chosen_preprocessing': chosen_tag,
		'unique_pred_classes': unique_classes,
		'indices': sel.tolist(),
		'pred_idx': None if pred_idx is None else pred_idx[:R],
		'conf': None if conf is None else conf[:R],
	}


__all__.extend([
	'visualize_test_images',
])


# =============================================
# Fashion-MNIST softmax training helper
# =============================================

def run_fashion_mnist_softmax(
	X: np.ndarray | None = None,
	y: np.ndarray | None = None,
	*,
	arch: tuple[int, ...] = (128, 64),
	lr: float = 1e-2,
	epochs: int = 10,
	batch_size: int = 128,
	optimizer: str = 'adam',
	seed: int = 42,
	test_size: float = 0.2,
	random_state: int = 42,
	stratify: bool = True,
	return_data: bool = True,
) -> dict:
	"""Fetch (if needed) Fashion-MNIST, train a softmax FFNN, and return metrics & splits."""
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import fetch_openml
	from sklearn.preprocessing import LabelEncoder

	# Load data if not provided
	label_encoder = None
	if X is None or y is None:
		fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
		X = fashion.data.astype(np.float32)
		y_raw = fashion.target
		label_encoder = LabelEncoder()
		y = label_encoder.fit_transform(y_raw)
	else:
		X = X.astype(np.float32)
		# y may be strings; encode if needed
		if not np.issubdtype(np.asarray(y).dtype, np.integer):
			label_encoder = LabelEncoder()
			y = label_encoder.fit_transform(np.asarray(y))

	# Scale to [0,1]
	if X.max() > 1.5:
		X = X / 255.0

	n_samples, n_features = X.shape
	n_classes = int(np.unique(y).size)

	# Split
	strat = y if stratify else None
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=float(test_size), random_state=int(random_state), stratify=strat
	)

	# One-hot targets
	Y_train = one_hot(y_train, n_classes)
	Y_test = one_hot(y_test, n_classes)

	# Build and train model
	layer_sizes = [n_features, *list(arch), n_classes]
	acts = ['relu'] * len(arch) + ['softmax']
	model = FFNN(
		layer_sizes=layer_sizes,
		activations=acts,
		loss='softmax',
		learning_rate=float(lr),
		optimizer=str(optimizer),
		reg=None,
		seed=int(seed),
	)
	_ = model.train(
		X_train, Y_train,
		epochs=int(epochs),
		batch_size=int(batch_size),
		verbose=False,
		print_every=1,
		shuffle=False,
		log_metrics=True,
	)

	# Evaluate
	metrics_train = model.evaluate(X_train, Y_train)
	metrics_test = model.evaluate(X_test, Y_test)

	result = {
		'model': model,
		'metrics_train': metrics_train,
		'metrics_test': metrics_test,
		'n_features': n_features,
		'n_classes': n_classes,
		'label_encoder': label_encoder,
	}
	if return_data:
		result.update({
			'X_train': X_train,
			'X_test': X_test,
			'y_train': y_train,
			'y_test': y_test,
			'Y_train': Y_train,
			'Y_test': Y_test,
		})
	return result


__all__.extend(['run_fashion_mnist_softmax'])

# =============================================
# Confusion matrix utility for classification
# =============================================

def confusion_matrix_plot(
	model: Any,
	X: np.ndarray,
	y_true: np.ndarray,
	*,
	label_encoder: Any | None = None,
	title: str = 'Confusion Matrix',
	cmap: str = 'Blues',
	normalize: str | None = 'true',
	annotate: str | None = 'auto',  # 'auto' | 'percent' | 'count' | None
) -> dict:
	"""Plot a confusion matrix (optionally normalized) and return its raw data + accuracy."""
	from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
	import matplotlib.pyplot as plt

	# Convert y_true to label indices
	if isinstance(y_true, np.ndarray) and y_true.ndim == 2 and y_true.shape[1] > 1:
		y_true_idx = np.argmax(y_true, axis=1)
	else:
		y_true_idx = np.asarray(y_true).astype(int).ravel()

	# Predictions
	def _pred_indices(m, Xb):
		try:
			pred = None
			if hasattr(m, 'predict'):
				pred = m.predict(Xb)
			elif hasattr(m, 'predict_proba'):
				pred = m.predict_proba(Xb)
			elif hasattr(m, 'forward'):
				pred = m.forward(Xb)
			pred = _to_numpy(pred)
			if pred.ndim == 2 and pred.shape[1] > 1:
				return np.argmax(pred, axis=1)
			return pred.astype(int).ravel()
		except Exception:
			return np.zeros(Xb.shape[0], dtype=int)

	y_pred_idx = _pred_indices(model, X)

	# Class names
	labels = None
	if label_encoder is not None:
		try:
			labels = list(label_encoder.classes_)
		except Exception:
			labels = None

	cm = confusion_matrix(y_true_idx, y_pred_idx)
	acc = accuracy_score(y_true_idx, y_pred_idx)

	# Decide what to display: counts or normalized
	cm_plot = cm.copy().astype(float)
	cm_norm = None
	if isinstance(normalize, str):
		norm_key = normalize.lower()
		if norm_key == 'true':  # normalize by row (true labels)
			row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
			cm_norm = cm / row_sums
			cm_plot = cm_norm
		elif norm_key == 'pred':  # normalize by column (pred labels)
			col_sums = np.clip(cm.sum(axis=0, keepdims=True), 1e-12, None)
			cm_norm = cm / col_sums
			cm_plot = cm_norm
		elif norm_key == 'all':  # normalize by total
			total = max(cm.sum(), 1)
			cm_norm = cm / total
			cm_plot = cm_norm

	fig, ax = plt.subplots(figsize=(7, 6))
	if labels is None:
		disp = ConfusionMatrixDisplay(cm_plot)
	else:
		disp = ConfusionMatrixDisplay(cm_plot, display_labels=labels)
	# Do not include default values to avoid clutter; we'll control annotations below
	disp.plot(cmap=cmap, ax=ax, colorbar=True, include_values=False)

	# Titles and axes labels
	ax.set_title(title + f" — acc={acc:.3f}")
	ax.set_xlabel('Predicted label')
	ax.set_ylabel('True label')

	# Rotate x tick labels for readability
	for tick in ax.get_xticklabels():
		tick.set_rotation(45)
		tick.set_ha('right')

	# Determine annotation mode
	ann_mode = annotate
	if ann_mode == 'auto':
		ann_mode = 'percent' if cm_norm is not None else 'count'

	# Add annotations (either counts or percentages), with contrast-aware color
	if ann_mode in ('percent', 'count'):
		data_for_text = cm_norm if (ann_mode == 'percent' and cm_norm is not None) else cm
		vmax = float(cm_plot.max()) if cm_plot.size else 1.0
		thresh = vmax / 2.0
		n_rows, n_cols = data_for_text.shape
		for i in range(n_rows):
			for j in range(n_cols):
				val = data_for_text[i, j]
				if ann_mode == 'percent' and cm_norm is not None:
					text_str = f"{val*100:.1f}%"
				else:
					text_str = f"{int(val)}"
				color = 'white' if cm_plot[i, j] > thresh else 'black'
				ax.text(j, i, text_str, ha='center', va='center', fontsize=9, color=color)

	plt.tight_layout(); plt.show()

	return {'cm': cm, 'cm_norm': cm_norm, 'accuracy': float(acc), 'labels': labels}

__all__.extend(['confusion_matrix_plot'])

# Final de-duplication of exports (preserve order of first occurrence)
__all__ = list(dict.fromkeys(__all__))

