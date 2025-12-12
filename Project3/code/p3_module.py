"""
Reusable utilities for Project 3: data loading, PCA+KMeans helpers,
seed control, and a small CNN with training/evaluation utilities.

Keep functions focused and composable so the notebook stays lean.
"""

import os
import random
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	roc_auc_score,
	confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42) -> None:
	"""Set global seeds for reproducibility (NumPy, Python, optional PyTorch)."""
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	try:
		import torch  # type: ignore

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		if hasattr(torch, "use_deterministic_algorithms"):
			torch.use_deterministic_algorithms(True)
	except Exception:
		# PyTorch not installed or deterministic setup failed; ignore gracefully
		pass


def load_split(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load images and labels for a given split from the cells dataset.

	Returns (X, y) where X has shape [N, 32, 32, 3] and y in {0,1}.
	"""
	images: list = []
	labels: list = []

	split_dir = os.path.join(data_dir, split)
	if not os.path.isdir(split_dir):
		raise FileNotFoundError(f"Split directory not found: {split_dir}")

	class_map = {"live": "Live_resized", "dead": "Dead_resized"}
	valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

	for label in ["live", "dead"]:
		folder = os.path.join(split_dir, class_map[label])
		if not os.path.isdir(folder):
			raise FileNotFoundError(f"Class folder not found: {folder}")
		files = [
			f
			for f in os.listdir(folder)
			if os.path.isfile(os.path.join(folder, f))
			and os.path.splitext(f)[1].lower() in valid_exts
		]
		for fname in sorted(files):
			path = os.path.join(folder, fname)
			img = Image.open(path).convert("RGB")
			if img.size != (32, 32):
				img = img.resize((32, 32))
			images.append(np.array(img))
			labels.append(0 if label == "live" else 1)
	return np.array(images), np.array(labels)


def fit_pca(X_train: np.ndarray, n_components: int = 3) -> Tuple[PCA, np.ndarray]:
	"""Fit PCA on flattened training images and return (pca, X_train_pca)."""
	X_flat = X_train.reshape(X_train.shape[0], -1)
	pca = PCA(n_components=n_components)
	X_train_pca = pca.fit_transform(X_flat)
	return pca, X_train_pca


def fit_pca_v2(X_train: np.ndarray, n_components: int = 3, standardize: bool = True) -> Tuple[PCA, np.ndarray]:
	"""Preprocess images (grayscale + normalize). Fit PCA on flattened training images and return (pca, X_train_pca)."""
	X_gray = np.dot(X_train[..., :3], [0.2989, 0.5870, 0.1140]) # Convert to grayscale - Luminosity Method
	if standardize: #standardize per pixel
		pixel_mean = X_gray.mean(axis=0, keepdims=True)
		pixel_std = X_gray.std(axis=0, keepdims=True) + 1e-8 # Avoid division by zero
		X_norm = (X_gray - pixel_mean) / pixel_std
	else:
		X_norm = X_gray / 255.0 # Scale to [0,1]
	
	X_flat = X_norm.reshape(X_norm.shape[0], -1)
	pca = PCA(n_components=n_components)
	X_train_pca = pca.fit_transform(X_flat)
	return pca, X_train_pca


def fit_kmeans(X_features: np.ndarray, n_clusters: int = 2, seed: int = 42) -> KMeans:
	"""Fit KMeans on feature matrix and return the estimator."""
	kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
	kmeans.fit(X_features)
	return kmeans


def pca_kmeans_predict(
	X: np.ndarray, pca: PCA, kmeans: KMeans, invert_labels: bool = True
) -> np.ndarray:
	"""
	Transform images with PCA and predict cluster labels; optional inversion to
	match semantic classes (live/dead) used in the notebook.
	"""
	X_flat = X.reshape(X.shape[0], -1)
	X_pca = pca.transform(X_flat)
	labels = kmeans.predict(X_pca)
	return (1 - labels) if invert_labels else labels


def pca_kmeans_predict_v2(
    X: np.ndarray, pca: PCA, kmeans: KMeans, invert_labels: bool = True
) -> np.ndarray:
    """
    Preprocess images (grayscale + scaling),
    transform with PCA, and predict cluster labels.
    """
    # Grayscale
    X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

	# Scale to [0,1]
    X_scaled = X_gray / 255.0

    # Flatten, PCA, k-means
    X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
    X_pca = pca.transform(X_flat)
    labels = kmeans.predict(X_pca)

    return (1 - labels) if invert_labels else labels




def plot_confusion(cm: np.ndarray, title: str) -> None:
	"""Plot a confusion matrix with consistent labels."""
	plt.figure(figsize=(6, 5))
	sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		cbar=False,
		xticklabels=["Predicted Live", "Predicted Dead"],
		yticklabels=["Actual Live", "Actual Dead"],
	)
	plt.title(title)
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	plt.show()


def build_mlp_pipeline(hidden: Tuple[int, int] = (64, 64), alpha: float = 1e-4, seed: int = 42) -> Pipeline:
	"""Create an sklearn Pipeline with StandardScaler + MLPClassifier."""
	return Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"mlp",
				MLPClassifier(
					hidden_layer_sizes=hidden,
					activation="relu",
					max_iter=300,
					alpha=alpha,
					random_state=seed,
				),
			),
		]
	)


# ---- Minimal CNN utilities (PyTorch) ----
try:
	import torch
	from torch import nn
	from torch.utils.data import DataLoader, TensorDataset
	import torch.optim as optim

	class CNN(nn.Module):
		def __init__(self, num_classes: int = 2):
			super().__init__()
			self.features = nn.Sequential(
				nn.Conv2d(3, 16, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2),  # 16x16
				nn.Conv2d(16, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(4),  # 4x4
			)
			self.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(32 * 4 * 4, 64),
				nn.ReLU(inplace=True),
				nn.Linear(64, num_classes),
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = self.features(x)
			x = self.classifier(x)
			return x

	def prepare_dataloaders(
		X_train: np.ndarray,
		y_train: np.ndarray,
		X_dev: np.ndarray,
		y_dev: np.ndarray,
		batch_size: int = 64,
	) -> Tuple[DataLoader, DataLoader]:
		"""Create PyTorch DataLoaders from numpy arrays (scale inputs to [0,1])."""
		X_train_t = torch.tensor(X_train).permute(0, 3, 1, 2).float() / 255.0
		X_dev_t = torch.tensor(X_dev).permute(0, 3, 1, 2).float() / 255.0
		y_train_t = torch.tensor(y_train).long()
		y_dev_t = torch.tensor(y_dev).long()

		train_dataset = TensorDataset(X_train_t, y_train_t)
		dev_dataset = TensorDataset(X_dev_t, y_dev_t)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
		return train_loader, dev_loader

	def train_cnn(
		model: nn.Module,
		train_loader: DataLoader,
		dev_loader: DataLoader,
		epochs: int = 10,
		device: Optional[torch.device] = None,
	) -> Dict[str, float]:
		"""Train CNN and return timing + last-epoch metrics."""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=1e-3)

		import time

		train_losses, dev_losses, train_accs, dev_accs = [], [], [], []
		overall_t0 = time.time()
		for _ in range(epochs):
			model.train()
			running_loss, correct, total = 0.0, 0, 0
			for xb, yb in train_loader:
				xb, yb = xb.to(device), yb.to(device)
				optimizer.zero_grad()
				logits = model(xb)
				loss = criterion(logits, yb)
				loss.backward()
				optimizer.step()
				running_loss += loss.item() * xb.size(0)
				preds = logits.argmax(dim=1)
				correct += (preds == yb).sum().item()
				total += yb.size(0)
			train_losses.append(running_loss / max(total, 1))
			train_accs.append(correct / max(total, 1))

			# Dev
			model.eval()
			dev_running_loss, dev_correct, dev_total = 0.0, 0, 0
			with torch.no_grad():
				for xb, yb in dev_loader:
					xb, yb = xb.to(device), yb.to(device)
					logits = model(xb)
					loss = criterion(logits, yb)
					dev_running_loss += loss.item() * xb.size(0)
					preds = logits.argmax(dim=1)
					dev_correct += (preds == yb).sum().item()
					dev_total += yb.size(0)
			dev_losses.append(dev_running_loss / max(dev_total, 1))
			dev_accs.append(dev_correct / max(dev_total, 1))

		total_train_time = time.time() - overall_t0
		return {
			"train_time_s": total_train_time,
			"last_train_loss": train_losses[-1] if train_losses else float("nan"),
			"last_dev_loss": dev_losses[-1] if dev_losses else float("nan"),
			"last_train_acc": train_accs[-1] if train_accs else float("nan"),
			"last_dev_acc": dev_accs[-1] if dev_accs else float("nan"),
		}

	def eval_dev_metrics(model: nn.Module, dev_loader: DataLoader, device: Optional[torch.device] = None) -> Dict[str, float]:
		"""Evaluate F1, ROC AUC, confusion matrix on the dev set."""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.eval()
		y_true, y_hat = [], []
		with torch.no_grad():
			for xb, yb in dev_loader:
				xb = xb.to(device)
				logits = model(xb)
				preds = logits.argmax(dim=1).cpu().numpy()
				y_hat.append(preds)
				y_true.append(yb.numpy())
		y_true = np.concatenate(y_true)
		y_hat = np.concatenate(y_hat)
		return {
			"f1": f1_score(y_true, y_hat),
			"roc_auc": roc_auc_score(y_true, y_hat),
			"confusion": confusion_matrix(y_true, y_hat),
		}

except Exception:
	# PyTorch may be unavailable; CNN helpers will not be defined.
	pass


