"""
Reusable utilities for Project 3: data loading, PCA+KMeans helpers,
seed control, and a small CNN with training/evaluation utilities.

Keep functions focused and composable so the notebook stays lean.
"""

import os
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

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
				# 32x32x16
				nn.ReLU(inplace=True),
				nn.MaxPool2d(2),  
				# 16x16x16
				nn.Conv2d(16, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				# 16x16x32
				nn.MaxPool2d(4),  
				# 4x4x32
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
		lr: float = 1e-3,
		device: Optional[torch.device] = None,
	) -> Dict[str, float]:
		"""Train CNN and return timing + last-epoch metrics."""
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=lr)

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
			# Full series for downstream plots/comparisons
			"train_losses": train_losses,
			"dev_losses": dev_losses,
			"train_accs": train_accs,
			"dev_accs": dev_accs,
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
	pass


# ---- PCA experiments helpers ----


def fit_pca_v2(X_train: np.ndarray, n_components: int = 3, standardize: bool = True) -> Tuple[PCA, np.ndarray]:
	"""Preprocess images (grayscale + normalize). Fit PCA on flattened training images and return (pca, X_train_pca)."""
	X_gray = rgb2gray(X_train)
	if standardize: #standardize per pixel (not used)
		pixel_mean = X_gray.mean(axis=0, keepdims=True)
		pixel_std = X_gray.std(axis=0, keepdims=True) + 1e-8 # Avoid division by zero
		X_norm = (X_gray - pixel_mean) / pixel_std
	else:
		X_norm = X_gray / 255.0 # Scale to [0,1]
	
	X_flat = X_norm.reshape(X_norm.shape[0], -1)
	pca = PCA(n_components=n_components)
	X_train_pca = pca.fit_transform(X_flat)
	return pca, X_train_pca


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


def rgb2gray(X): # Convert to grayscale - Luminosity Method
    Xg = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])
    return Xg / 255.0 if Xg.max() > 1.5 else Xg


def conv2_same(img, k): # manual convolution replaced by scipy convolve
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(pad[i:i+kh, j:j+kw] * k)
    return out


def extract_cell_features(
    X,
    center_r=2.5,
    ring_r1=3.5,
    ring_r2=7.0,
    dark_thr=0.30,
):
    """
    Extract hand-crafted features from cell images:
	1. gmean: Global mean intensity
	2. gstd: Global intensity standard deviation
	3. cmean: Mean intensity in center region
	4. rmean: Mean intensity in surrounding ring
	5. c_minus_r: Center-ring intensity difference
	6. c_over_r: Center-ring intensity ratio
	7. radial_std: Standard deviation of radial intensity profile
	8. edge_energy: edge energy (gradient magnitude mean)
	9. lap_var: Laplacian variance (sharpness measure)
	10. dark_frac: Fraction of dark pixels
	11. center_p90: 90th percentile intensity in center region
	12. peak_over_ring: Center peak-to-ring intensity ratio

	
    Returns: (N, 12) feature matrix
    """
    imgs = rgb2gray(X).astype(float)
    N, H, W = imgs.shape

    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    center = r <= center_r
    ring   = (r > ring_r1) & (r <= ring_r2)

	# global features for all images
    gmean_all = imgs.mean(axis=(1, 2))
    gstd_all  = imgs.std(axis=(1, 2))

    cmean_all = imgs[:, center].mean(axis=1)
    rmean_all = imgs[:, ring].mean(axis=1)

    dark_frac_all = (imgs < dark_thr).mean(axis=(1, 2))
    center_p90_all = np.percentile(imgs[:, center], 90, axis=1)

    rbins = np.clip(r.astype(int), 0, int(r.max()))
    nbins = rbins.max() + 1

    sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float) / 4.0
    sy = sx.T
    lap = np.array([[0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0]], dtype=float)

    feats = []
    eps = 1e-8

    for i, img in enumerate(imgs):
        gmean = gmean_all[i]
        gstd  = gstd_all[i]

        cmean = cmean_all[i]
        rmean = rmean_all[i]

        c_minus_r = cmean - rmean
        c_over_r  = (cmean + eps) / (rmean + eps)

        center_p90 = center_p90_all[i]
        peak_over_ring = (center_p90 + eps) / (rmean + eps)

        # Radial intensity profile
        prof = np.array([img[rbins == k].mean() for k in range(nbins)])
        radial_std = prof.std()

        # Edge / sharpness features (C-optimized)
        gx = convolve(img, sx, mode="reflect")
        gy = convolve(img, sy, mode="reflect")
        edge_energy = np.mean(np.sqrt(gx*gx + gy*gy))

        l = convolve(img, lap, mode="reflect")
        lap_var = l.var()

        dark_frac = dark_frac_all[i]

        feats.append([
            gmean, gstd,
            cmean, rmean,
            c_minus_r, c_over_r,
            radial_std, edge_energy, lap_var,
            dark_frac,
            center_p90, peak_over_ring
        ])

    return np.asarray(feats, dtype=float)


def best_flip(y_true, y_pred):
    f1_id = f1_score(y_true, y_pred)
    f1_fl = f1_score(y_true, 1 - y_pred)
    return (1 - y_pred) if f1_fl > f1_id else y_pred


# ---- PCA + KMeans on engineered features ----
def fit_feature_pca_kmeans(
    X_train: np.ndarray,
    n_components: int = 3,
    seed: int = 42,
):
    """
    Fit StandardScaler + PCA + KMeans on engineered features.
    """
    F_train = extract_cell_features(X_train)

    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)

    pca = PCA(n_components=n_components, random_state=seed)
    Z_train = pca.fit_transform(F_train_s)

    kmeans = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=50
    )
    kmeans.fit(Z_train)

    return scaler, pca, kmeans


def predict_feature_pca_kmeans(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    kmeans: KMeans,
):
    F = extract_cell_features(X)
    F_s = scaler.transform(F)
    Z = pca.transform(F_s)
    return kmeans.predict(Z)