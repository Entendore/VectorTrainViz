import os, copy, random, warnings, sys, traceback
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('QtAgg')  # Ensure Matplotlib uses Qt backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # Moved to top level to prevent AttributeError
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QTabWidget, QFormLayout, 
                               QSpinBox, QDoubleSpinBox, QPlainTextEdit, QLabel, 
                               QCheckBox, QScrollArea, QComboBox, QStackedWidget, 
                               QLineEdit)
from PySide6.QtCore import QThread, Signal, Qt

# --- optional vmap for batched loss-landscape ---
try:
    from torch.func import functional_call as _func_call, vmap as _vmap
    _HAS_VMAP = True
except (ImportError, AttributeError):
    _HAS_VMAP = False

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(x, **kw): return x

warnings.filterwarnings('ignore')

# ==================================================================
# 0. Reproducibility & Config
# ==================================================================
SEED = 42
DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    epochs:          int  = 60
    batch_size:      int  = 32
    lr:              float = 1e-2
    weight_decay:    float = 0.0
    patience:        int  = 15
    snapshot_every:  int  = 1
    ckpt_every:      int  = 6
    use_scheduler:   bool = True
    swa_last_k:      int  = 10

# ==================================================================
# 1. Lightweight Vector Database (Optimized)
# ==================================================================
class SimpleVectorDB:
    def __init__(self, name="default", distance="cosine"):
        self.name = name
        self.distance = distance
        self.embeddings = []
        self.metadatas = []
        self.documents = []
        self.ids = []
        self._nbrs = None
        self._X = None

    def upsert(self, embeddings, documents, metadatas, ids):
        self.embeddings.append(np.array(embeddings[0]))
        self.documents.append(documents[0])
        self.metadatas.append(metadatas[0])
        self.ids.append(ids[0])
        self._nbrs = None
        self._X = None

    def _fit_index(self):
        if not self.embeddings: return
        self._X = np.array(self.embeddings)
        metric = "cosine" if self.distance == "cosine" else "euclidean"
        self._nbrs = NearestNeighbors(n_neighbors=len(self._X), metric=metric).fit(self._X)

    def query(self, query_embeddings, n_results=5, include=None):
        if not self.embeddings:
            return {"metadatas": [[]], "distances": [[]]}
        if self._nbrs is None:
            self._fit_index()
            
        q = np.array(query_embeddings[0]).reshape(1, -1)
        k = min(n_results, len(self._X))
        distances, indices = self._nbrs.kneighbors(q, n_neighbors=k)
        return {
            "metadatas": [[self.metadatas[i] for i in indices[0]]],
            "distances": [distances[0].tolist()]
        }

def setup_vector_db(name: str, persist: bool = False, distance: str = "cosine") -> SimpleVectorDB:
    return SimpleVectorDB(name=name, distance=distance)

# ==================================================================
# 2. Dataset Preparation
# ==================================================================
def prepare_moons_data() -> Tuple[torch.Tensor, ...]:
    X, y = make_moons(n_samples=400, noise=0.15, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    return (torch.FloatTensor(Xtr), torch.FloatTensor(Xte),
            torch.FloatTensor(ytr).view(-1,1), torch.FloatTensor(yte).view(-1,1))

def prepare_digits_data() -> Tuple[torch.Tensor, ...]:
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.2, random_state=SEED)
    return (torch.FloatTensor(Xtr), torch.FloatTensor(Xte),
            torch.LongTensor(ytr), torch.LongTensor(yte))

# ==================================================================
# 3. Model
# ==================================================================
class MLP(nn.Module):
    def __init__(self, in_dim:int, hidden:List[int], out_dim:int,
                 dropout:float=0.0, binary:bool=False):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        if binary:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def load_flat_params(self, flat: torch.Tensor) -> None:
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p).to(p.device))
            idx += n

    def first_layer_weights(self) -> torch.Tensor:
        for m in self.net:
            if isinstance(m, nn.Linear):
                return m.weight.detach().cpu()
        raise ValueError("No Linear layer")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def per_layer_grad_norms(self) -> Dict[str, float]:
        return {n: p.grad.detach().norm(2).item()
                for n, p in self.named_parameters() if p.grad is not None}

    def param_shapes(self) -> List[Tuple[str, torch.Size, int]]:
        return [(n, p.shape, p.numel()) for n, p in self.named_parameters()]

    def get_linear_layer_names(self) -> List[str]:
        return [name for name, m in self.named_modules() if isinstance(m, nn.Linear)]

# ==================================================================
# 4. Training
# ==================================================================
@dataclass
class TrainingHistory:
    weight_snapshots:  List[np.ndarray]            = field(default_factory=list)
    layer_weight_snapshots: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    
    # Algorithmic Layer Groupings
    group_definitions: Dict[str, Dict]             = field(default_factory=dict)
    grouped_weight_snapshots: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    
    # Vector Segments (Independent of Layers)
    segment_definitions: Dict[str, Any]            = field(default_factory=dict)
    segment_weight_snapshots: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    segment_grad_norms: Dict[str, List[float]]      = field(default_factory=dict)
    
    train_losses:      List[float]                 = field(default_factory=list)
    train_accs:        List[float]                 = field(default_factory=list)
    val_losses:        List[float]                 = field(default_factory=list)
    val_accs:          List[float]                 = field(default_factory=list)
    grad_norms:        List[float]                 = field(default_factory=list)
    per_layer_grads:   List[Dict[str, float]]      = field(default_factory=list)
    lrs:               List[float]                 = field(default_factory=list)
    epochs:            List[int]                   = field(default_factory=list)
    model_snapshots:   List[Dict]                  = field(default_factory=list)
    snapshot_epochs:   List[int]                   = field(default_factory=list)
    snapshot_val_accs: List[float]                 = field(default_factory=list)
    dead_relu_stats:   List[Dict[str, float]]      = field(default_factory=list)
    swa_weights:       Optional[np.ndarray]        = None
    swa_val_acc:       float                       = 0.0
    hessian_eig:       float                       = 0.0
    best_state:        Optional[Dict]              = None
    best_val_acc:      float                       = -1.0
    best_epoch:        int                         = -1

def _count_dead_relu(model: nn.Module, X: torch.Tensor) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            def make_hook(n):
                def hook(mod, inp, out):
                    stats[n] = (out == 0).float().mean().item()
                return hook
            handles.append(m.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(X[:200].to(DEVICE))
    for h in handles:
        h.remove()
    return stats

def _evaluate(model, loader, criterion, binary):
    model.eval(); total=correct=0; loss_sum=0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss_sum += criterion(out, yb).item()*xb.size(0)
            correct += ((out>0.5).float()==yb).sum().item() if binary \
                       else (out.argmax(1)==yb).sum().item()
            total += xb.size(0)
    return loss_sum/total, correct/total

def train_and_vectorize(model, X_train, y_train, X_test, y_test,
                         collection, dataset_name, cfg, binary, stop_check=None):
    model = model.to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.BCELoss() if binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
                 if cfg.use_scheduler else None)

    h = TrainingHistory()
    patience_left = cfg.patience
    
    layer_names = model.get_linear_layer_names()
    for ln in layer_names:
        h.layer_weight_snapshots[ln] = []

    print(f"\nTraining {dataset_name}: {model.num_params()} params "
          f"on {DEVICE}  ({len(train_loader)} batches/epoch)")

    pbar = tqdm(range(cfg.epochs), desc=f"{dataset_name:>6s}", disable=not _HAS_TQDM)
    for epoch in pbar:
        if stop_check and stop_check():
            print("Training stopped by user.")
            break
            
        model.train()
        ep_loss = ep_correct = ep_total = 0
        for xb, yb in train_loader:
            if stop_check and stop_check(): break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out  = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.sqrt(
                sum((p.grad.detach()**2).sum()
                    for p in model.parameters() if p.grad is not None)).item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = xb.size(0)
            ep_loss    += loss.item()*bs
            ep_correct += ((out>0.5).float()==yb).sum().item() if binary \
                          else (out.argmax(1)==yb).sum().item()
            ep_total   += bs

        tr_loss = ep_loss/ep_total
        tr_acc  = ep_correct/ep_total
        val_loss, val_acc = _evaluate(model, test_loader, criterion, binary)
        if scheduler: scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        if val_acc > h.best_val_acc:
            h.best_val_acc = val_acc
            h.best_epoch   = epoch
            h.best_state    = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
            patience_left = cfg.patience
        elif cfg.patience > 0:
            patience_left -= 1

        if epoch % cfg.snapshot_every == 0 or epoch == cfg.epochs-1:
            flat = model.flat_params().cpu().numpy()
            dead = _count_dead_relu(model, X_train)
            collection.upsert(
                embeddings=[flat.tolist()],
                documents=[f"{dataset_name} epoch {epoch}"],
                metadatas=[{"epoch":epoch, "loss":float(tr_loss), "train_acc":float(tr_acc),
                             "val_acc":float(val_acc), "grad_norm":float(grad_norm),
                             "lr":float(lr_now), "dataset":dataset_name,
                             "dead_relu_pct":float(np.mean(list(dead.values())))}],
                ids=[f"{dataset_name}_epoch_{epoch}"],
            )
            h.weight_snapshots.append(flat)
            h.train_losses.append(tr_loss)
            h.train_accs.append(tr_acc)
            h.val_losses.append(val_loss)
            h.val_accs.append(val_acc)
            h.grad_norms.append(grad_norm)
            h.per_layer_grads.append(model.per_layer_grad_norms())
            h.lrs.append(lr_now)
            h.epochs.append(epoch)
            h.dead_relu_stats.append(dead)
            
            for ln in layer_names:
                layer_mod = model.get_submodule(ln)
                w = layer_mod.weight.detach().cpu().numpy().flatten()
                b = layer_mod.bias.detach().cpu().numpy().flatten()
                h.layer_weight_snapshots[ln].append(np.concatenate([w, b]))

            # --- Algorithmic Layer Groupings ---
            n_layers = len(layer_names)
            half = n_layers // 2
            groups = {
                "Group: All Weights": {"type": "weights_only", "layers": layer_names},
                "Group: All Biases": {"type": "biases_only", "layers": layer_names},
                "Group: Early Layers (First Half)": {"type": "full", "layers": layer_names[:half]},
                "Group: Late Layers (Second Half)": {"type": "full", "layers": layer_names[half:]},
                "Group: Feature Extractor (Excl. Last)": {"type": "full", "layers": layer_names[:-1]},
                "Group: Output Layer": {"type": "full", "layers": layer_names[-1:] if layer_names else []},
                "Group: Even Layers": {"type": "full", "layers": layer_names[0::2]},
                "Group: Odd Layers": {"type": "full", "layers": layer_names[1::2]},
                "Group: Hidden Core (Excl. First & Last)": {"type": "full", "layers": layer_names[1:-1]},
            }
            h.group_definitions = groups
            
            for gname, gdef in groups.items():
                if not gdef["layers"]: continue
                w_list = []
                b_list = []
                for ln in gdef["layers"]:
                    lm = model.get_submodule(ln)
                    w_list.append(lm.weight.detach().cpu().numpy().flatten())
                    b_list.append(lm.bias.detach().cpu().numpy().flatten())
                
                if gdef["type"] == "weights_only":
                    snap = np.concatenate(w_list) if w_list else np.array([])
                elif gdef["type"] == "biases_only":
                    snap = np.concatenate(b_list) if b_list else np.array([])
                else: # full
                    snap = np.concatenate([np.concatenate(w_list), np.concatenate(b_list)]) if w_list else np.array([])
                
                if gname not in h.grouped_weight_snapshots:
                    h.grouped_weight_snapshots[gname] = []
                h.grouped_weight_snapshots[gname].append(snap)

            # --- Vector Segments (Independent of Layers) ---
            n_params = model.num_params()
            if not h.segment_definitions:
                q1 = n_params // 4
                q2 = n_params // 2
                q3 = 3 * n_params // 4
                
                h.segment_definitions = {
                    "Segment: Bottom 50% Params": (0, q2),
                    "Segment: Top 50% Params": (q2, n_params),
                    "Segment: First Quartile (0-25%)": (0, q1),
                    "Segment: Second Quartile (25-50%)": (q1, q2),
                    "Segment: Third Quartile (50-75%)": (q2, q3),
                    "Segment: Fourth Quartile (75-100%)": (q3, n_params),
                }
                
                rng = np.random.default_rng(SEED)
                rand_size = max(10, n_params // 4)
                h.segment_definitions["Segment: Random 25% Subset"] = rng.choice(n_params, size=rand_size, replace=False)
            
            for seg_name, seg_def in h.segment_definitions.items():
                if seg_name not in h.segment_weight_snapshots:
                    h.segment_weight_snapshots[seg_name] = []
                    h.segment_grad_norms[seg_name] = []
                    
                if isinstance(seg_def, tuple):
                    start, end = seg_def
                    h.segment_weight_snapshots[seg_name].append(flat[start:end].copy())
                else:
                    h.segment_weight_snapshots[seg_name].append(flat[seg_def].copy())
                
                flat_grad = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).cpu().numpy()
                if isinstance(seg_def, tuple):
                    seg_grad = flat_grad[start:end]
                else:
                    seg_grad = flat_grad[seg_def]
                h.segment_grad_norms[seg_name].append(np.linalg.norm(seg_grad))

        if epoch % cfg.ckpt_every == 0 or epoch == cfg.epochs-1:
            h.model_snapshots.append({k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()})
            h.snapshot_epochs.append(epoch)
            h.snapshot_val_accs.append(val_acc)

        if _HAS_TQDM:
            pbar.set_postfix(loss=f"{tr_loss:.4f}", vacc=f"{val_acc:.4f}")

        if cfg.patience>0 and patience_left<=0:
            print(f"  ⏹ Early stopping at epoch {epoch+1} "
                  f"(best val_acc={h.best_val_acc:.4f} @ ep {h.best_epoch+1})")
            break

    if h.best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in h.best_state.items()})
        print(f"  ✓ Restored best model (val_acc={h.best_val_acc:.4f} @ epoch {h.best_epoch+1})")

    if cfg.swa_last_k > 0 and len(h.weight_snapshots) >= cfg.swa_last_k:
        snaps = h.weight_snapshots[-cfg.swa_last_k:]
        h.swa_weights = np.mean(snaps, axis=0)
        model.load_flat_params(torch.tensor(h.swa_weights))
        _, h.swa_val_acc = _evaluate(model, test_loader, criterion, binary)
        print(f"  ✓ SWA (last {cfg.swa_last_k} snapshots): val_acc={h.swa_val_acc:.4f}")

    h.hessian_eig = _estimate_hessian_eig(model, X_test, y_test, criterion, binary)
    print(f"  ✓ Hessian λ_max ≈ {h.hessian_eig:.2f} "
          f"({'sharp' if h.hessian_eig>1 else 'flat'} minimum)")

    return h

# ==================================================================
# 5. Analysis
# ==================================================================
def _estimate_hessian_eig(model, X, y, criterion, binary, n_iter=15, subsample=200) -> float:
    model.eval()
    Xs = X[:subsample].to(DEVICE)
    ys = y[:subsample].to(DEVICE)
    n  = model.num_params()
    v  = torch.randn(n, device=DEVICE)
    v  = v / v.norm()
    eig = 0.0
    for _ in range(n_iter):
        out  = model(Xs)
        loss = criterion(out, ys)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.flatten() for g in grads])
        hvp = torch.autograd.grad(flat_grads, model.parameters(), grad_outputs=v)
        flat_hvp = torch.cat([g.flatten() for g in hvp])
        
        if torch.isnan(flat_hvp).any() or torch.isinf(flat_hvp).any():
            print("  ⚠ NaN/Inf in HVP. Returning 0.")
            return 0.0
            
        eig = (v * flat_hvp).sum().item()
        norm = flat_hvp.norm()
        if norm < 1e-10: return 0.0
        v = flat_hvp / norm
    return eig

def _filter_normalized_direction(model, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    parts = []
    for _, p in model.named_parameters():
        d = torch.randn(p.shape, generator=g)
        p_norm = p.detach().norm()
        d_norm = d.norm()
        if d_norm > 0: d = d * (p_norm / d_norm)
        parts.append(d.flatten())
    return torch.cat(parts)

# ==================================================================
# 6. Visualization (Returns Matplotlib Figures for GUI)
# ==================================================================
def _pca_3d(snapshots):
    S = np.stack(snapshots)
    n_comp = min(3, max(1, S.shape[0] - 1))
    if S.shape[0] < 2: return np.zeros((S.shape[0], 3)), None
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(S)
    if transformed.shape[1] < 3:
        transformed = np.pad(transformed, ((0,0), (0, 3 - transformed.shape[1])))
    return transformed, pca

def visualize_trajectory(history, dataset_name, color_metric="epoch"):
    coords, pca = _pca_3d(history.weight_snapshots)
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    metric_map = {"epoch":history.epochs, "loss":history.train_losses,
                  "val_acc":history.val_accs, "grad_norm":history.grad_norms}
    cvals = metric_map.get(color_metric, history.epochs)
    ctitle = {"epoch":"Epoch","loss":"Train Loss","val_acc":"Val Acc",
              "grad_norm":"Grad Norm"}[color_metric]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    img = ax.scatter(x, y, z, c=cvals, cmap='viridis', s=40, label='Trajectory')
    ax.plot(x, y, z, color='gray', alpha=0.5, lw=1.5)
    fig.colorbar(img, ax=ax, label=ctitle)
    
    if history.swa_weights is not None and pca is not None:
        swa_coord = pca.transform(history.swa_weights.reshape(1, -1))[0]
        ax.scatter([swa_coord[0]], [swa_coord[1]], [swa_coord[2]], 
                   color='gold', s=120, marker='D', edgecolors='black', label='SWA')

    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, marker='D', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, marker='D', label='End')
    
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(f"{dataset_name} - Trajectory ({ctitle})")
    ax.legend()
    return fig

def visualize_metrics(history, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    e = history.epochs

    axes[0,0].plot(e, history.train_losses, lw=2, label='train')
    axes[0,0].plot(e, history.val_losses, lw=2, ls='--', label='val')
    axes[0,0].axvline(history.best_epoch, color='grey', ls=':', alpha=0.5, label=f'best (ep {history.best_epoch+1})')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss'); axes[0,0].legend()
    axes[0,0].set_title(f'{dataset_name} — Loss'); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(e, history.train_accs, lw=2, label='train')
    axes[0,1].plot(e, history.val_accs, lw=2, ls='--', label='val')
    axes[0,1].axvline(history.best_epoch, color='grey', ls=':', alpha=0.5)
    if history.swa_val_acc:
        axes[0,1].axhline(history.swa_val_acc, color='gold', ls='-.', alpha=0.7, label=f"SWA ({history.swa_val_acc:.4f})")
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend(); axes[0,1].set_title(f'{dataset_name} — Accuracy')
    axes[0,1].grid(alpha=0.3)

    if history.per_layer_grads:
        layer_names = list(history.per_layer_grads[0].keys())
        for ln in layer_names:
            vals = [d.get(ln, 0) for d in history.per_layer_grads]
            short = (ln.replace('net.0.weight','L1-w').replace('net.0.bias','L1-b')
                      .replace('net.3.weight','L2-w').replace('net.3.bias','L2-b')
                      .replace('net.6.weight','L3-w').replace('net.6.bias','L3-b')
                      .replace('net.9.weight','L4-w').replace('net.9.bias','L4-b'))
            axes[1,0].plot(e, vals, lw=1.5, label=short)
        axes[1,0].set_yscale('log'); axes[1,0].legend(fontsize=7, ncol=2)
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('‖grad‖₂')
    axes[1,0].set_title(f'{dataset_name} — Per-layer Gradient Norms')
    axes[1,0].grid(alpha=0.3)

    if history.dead_relu_stats:
        relu_names = list(history.dead_relu_stats[0].keys())
        for i, rn in enumerate(relu_names):
            vals = [d.get(rn, 0) for d in history.dead_relu_stats]
            axes[1,1].plot(e, vals, lw=2, label=f'ReLU {i+1}')
        axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Dead fraction')
        axes[1,1].set_title(f'{dataset_name} — Dead ReLU Fraction')
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    
    fig.tight_layout()
    return fig

def visualize_global_grad_norm(history, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epochs, history.grad_norms, lw=2, label='Total Grad Norm', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (L2)')
    ax.set_title(f'{dataset_name} — Global Gradient Norm')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def visualize_weight_heatmap(history, dataset_name, n_bins=80):
    snaps = history.weight_snapshots
    all_w = np.concatenate(snaps)
    w_min, w_max = all_w.min(), all_w.max()
    hist = np.zeros((len(snaps), n_bins))
    for i, s in enumerate(snaps):
        hist[i] = np.histogram(s, bins=n_bins, range=(w_min, w_max))[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(hist, aspect='auto', cmap='magma',
                   extent=[w_min, w_max, len(snaps), 0], origin='upper')
    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Weight value'); ax.set_ylabel('Snapshot idx')
    ax.set_title(f'{dataset_name} — Weight distribution evolution')
    fig.tight_layout()
    return fig

def visualize_layer_shift(history, dataset_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_name, snaps in history.layer_weight_snapshots.items():
        init_weights = snaps[0]
        distances = [np.linalg.norm(s - init_weights) for s in snaps]
        ax.plot(history.epochs, distances, lw=2, label=layer_name, marker='o', markersize=3)
    
    ax.set_title(f"{dataset_name} — Layer Distance from Initialization")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Distance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def visualize_pca_variance(history, dataset_name):
    snaps = np.stack(history.weight_snapshots)
    n_comp = min(10, snaps.shape[0]-1)
    if n_comp < 1: return None
    pca = PCA(n_components=n_comp).fit(snaps)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, n_comp+1), pca.explained_variance_ratio_, color='skyblue', edgecolor='black')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"{dataset_name} — PCA Explained Variance (Up to 10 PCs)")
    ax.set_xticks(range(1, n_comp+1))
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        
    fig.tight_layout()
    return fig

def visualize_lr_loss(history, dataset_name):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(history.epochs, history.train_losses, color=color, lw=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)  
    ax2.plot(history.epochs, history.lrs, color=color, lw=2, ls='--', label='LR')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(f"{dataset_name} — Learning Rate vs. Loss")
    fig.tight_layout()
    return fig

def visualize_receptive_fields(model, dataset_name, n_units=16):
    w = model.first_layer_weights()
    n_show = min(n_units, w.shape[0])
    cols = 8
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5), squeeze=False)
    vmax = w.abs().max()
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        if i < n_show:
            ax.imshow(w[i].reshape(8,8), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f"u{i}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{dataset_name} — First-layer receptive fields", y=1.02)
    fig.tight_layout()
    return fig

def animate_decision_boundary(model, history, X, y, dataset_name, grid_steps=150):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, grid_steps)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, grid_steps)
    gx, gy = np.meshgrid(xs, ys)
    grid = torch.FloatTensor(np.c_[gx.ravel(), gy.ravel()])

    snaps = history.model_snapshots
    epochs = history.snapshot_epochs
    vaccs = history.snapshot_val_accs
    if not snaps: return None, None

    def update(i):
        ax.cla()
        state_dict = {k: v.to(DEVICE) for k, v in snaps[i].items()}
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            pred = model(grid.to(DEVICE)).cpu().numpy().reshape(gx.shape)
        ax.contourf(gx, gy, pred, levels=50, cmap='RdBu_r', vmin=0, vmax=1, alpha=0.5)
        ax.contour(gx, gy, pred, levels=[0.5], colors='black', linewidths=2)
        ax.scatter(X[:,0], X[:,1], c=y.squeeze(), s=12, cmap='bwr', edgecolors='k', linewidths=0.3)
        ax.set_title(f"{dataset_name} — Epoch {epochs[i]}  (val_acc={vaccs[i]:.3f})")
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=400, blit=False)
    fig.tight_layout()
    return fig, ani

def visualize_sigmoid_confidence(model, X, y, dataset_name, grid_steps=200):
    model.eval()
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, grid_steps)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, grid_steps)
    gx, gy = np.meshgrid(xs, ys)
    grid = torch.FloatTensor(np.c_[gx.ravel(), gy.ravel()]).to(DEVICE)
    with torch.no_grad():
        prob = model(grid).cpu().numpy().reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(gx, gy, prob, levels=50, cmap='RdBu_r', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='P(class = 1)')
    ax.contour(gx, gy, prob, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[:,0], X[:,1], c=y.squeeze(), s=15, cmap='bwr', edgecolors='k', linewidths=0.3)
    ax.set_title(f'{dataset_name} — Sigmoid confidence surface')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    fig.tight_layout()
    return fig

def visualize_loss_landscape(model, history, X, y, dataset_name, binary, grid_size=30, subsample=200, use_pca_directions=True):
    w_final_np = history.weight_snapshots[-1]
    w_final = torch.tensor(w_final_np, dtype=torch.float32, device=DEVICE)
    model = model.to(DEVICE)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(subsample, len(X)), replace=False)
    Xs = X[idx].to(DEVICE)
    ys = y[idx].to(DEVICE)
    criterion = nn.BCELoss() if binary else nn.CrossEntropyLoss()

    if use_pca_directions:
        S = np.stack(history.weight_snapshots)
        pca = PCA(n_components=min(2, S.shape[0]-1)).fit(S)
        if pca.n_components_ < 2: return None
        d1 = torch.tensor(pca.components_[0], dtype=torch.float32, device=DEVICE)
        d2 = torch.tensor(pca.components_[1], dtype=torch.float32, device=DEVICE)
        scale1 = np.sqrt(pca.explained_variance_[0]) * 2.0
        scale2 = np.sqrt(pca.explained_variance_[1]) * 2.0
        dir_label = "PCA"
    else:
        d1 = _filter_normalized_direction(model, seed=0).to(DEVICE)
        d2 = _filter_normalized_direction(model, seed=1).to(DEVICE)
        scale1 = scale2 = 0.5 * w_final.norm().item()
        dir_label = "random (filter-norm)"

    alphas = np.linspace(-scale1, scale1, grid_size)
    betas  = np.linspace(-scale2, scale2, grid_size)

    if _HAS_VMAP:
        losses = _loss_landscape_vmap(model, Xs, ys, criterion, w_final, d1, d2, alphas, betas)
    else:
        print("  ⚠ vmap unavailable. Using slow loop for loss landscape...")
        losses = _loss_landscape_loop(model, Xs, ys, criterion, w_final, d1, d2, alphas, betas)

    snaps = history.weight_snapshots
    if use_pca_directions:
        traj_2d = pca.transform(snaps)
        traj_a, traj_b = traj_2d[:, 0], traj_2d[:, 1]
    else:
        w_diff = snaps - w_final_np
        traj_a = w_diff @ d1.cpu().numpy()
        traj_b = w_diff @ d2.cpu().numpy()

    swa_a = swa_b = None
    if history.swa_weights is not None:
        if use_pca_directions:
            swa_2d = pca.transform(history.swa_weights.reshape(1, -1))
            swa_a, swa_b = swa_2d[0, 0], swa_2d[0, 1]
        else:
            d = (history.swa_weights - w_final_np)
            swa_a = d @ d1.cpu().numpy()
            swa_b = d @ d2.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(alphas, betas, losses, levels=30, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Loss')
    ax.plot(traj_a, traj_b, 'w.-', lw=2, ms=6, label='Trajectory')
    ax.plot(traj_a[0], traj_b[0], 'go', ms=12, label='Start')
    ax.plot(traj_a[-1], traj_b[-1], 'r*', ms=15, label='End')
    if swa_a is not None:
        ax.plot(swa_a, swa_b, 'y*', ms=18, label=f'SWA ({history.swa_val_acc:.4f})')
    ax.set_xlabel(f'Direction 1 ({dir_label})')
    ax.set_ylabel(f'Direction 2 ({dir_label})')
    ax.set_title(f"{dataset_name} — Loss Landscape ({dir_label})")
    ax.legend()
    fig.tight_layout()
    return fig

def _loss_landscape_vmap(model, Xs, ys, criterion, w_final, d1, d2, alphas, betas):
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    G = A.size
    grid_w = (w_final.unsqueeze(0)
              + torch.tensor(A.ravel(), device=DEVICE).unsqueeze(1) * d1.unsqueeze(0)
              + torch.tensor(B.ravel(), device=DEVICE).unsqueeze(1) * d2.unsqueeze(0))

    param_info = model.param_shapes()
    batched_params = {}
    idx = 0
    for name, shape, numel in param_info:
        batched_params[name] = grid_w[:, idx:idx+numel].view(G, *shape).to(DEVICE)
        idx += numel

    def forward_loss(params):
        out = _func_call(model, params, Xs)
        return criterion(out, ys)

    losses = _vmap(forward_loss, in_dims=(0,))(batched_params)
    return losses.view(len(alphas), len(betas)).cpu().numpy()

def _loss_landscape_loop(model, Xs, ys, criterion, w_final, d1, d2, alphas, betas):
    losses = np.zeros((len(alphas), len(betas)))
    orig_state = copy.deepcopy(model.state_dict())
    model.eval()
    for i, a in enumerate(tqdm(alphas, disable=not _HAS_TQDM)):
        for j, b in enumerate(betas):
            model.load_flat_params(w_final + a*d1 + b*d2)
            with torch.no_grad():
                losses[i, j] = criterion(model(Xs), ys).item()
    model.load_state_dict(orig_state)
    return losses

def visualize_confusion_matrix(model, X_test, y_test, dataset_name):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).argmax(1).cpu().numpy()
    cm = confusion_matrix(y_test.numpy(), preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')
    ax.set_title(f'{dataset_name} — Confusion Matrix (normalised)')
    fig.tight_layout()
    return fig

# ==================================================================
# 6.1 Sectioned Visualizations
# ==================================================================
def visualize_combined_sectioned_trajectory(history, dataset_name):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.colormaps['tab10']
    
    for i, (layer_name, snaps) in enumerate(history.layer_weight_snapshots.items()):
        if len(snaps) < 2: continue
            
        S = np.stack(snaps)
        pca = PCA(n_components=min(3, S.shape[0]-1))
        transformed = pca.fit_transform(S)
        
        if transformed.shape[1] < 3:
            transformed = np.pad(transformed, ((0,0), (0, 3 - transformed.shape[1])))
            
        max_val = np.max(np.abs(transformed))
        if max_val > 1e-10:
            transformed = transformed / max_val
            
        transformed = transformed - transformed[0]
        
        x, y, z = transformed[:,0], transformed[:,1], transformed[:,2]
        color = cmap(i)
        
        ax.plot(x, y, z, color=color, lw=2, label=layer_name, marker='o', markersize=3)
        ax.scatter([x[0]], [y[0]], [z[0]], color=color, s=80, marker='o', edgecolors='k')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=120, marker='D', edgecolors='k')
        
    ax.set_title(f"{dataset_name} - All Layers Combined Trajectory (Normalized)")
    ax.set_xlabel('PC1 (norm)'); ax.set_ylabel('PC2 (norm)'); ax.set_zlabel('PC3 (norm)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def _get_section_snaps(history, section_name):
    """Helper to fetch snapshots for layers, groups, or segments."""
    if section_name in history.layer_weight_snapshots:
        return history.layer_weight_snapshots[section_name]
    elif section_name in history.grouped_weight_snapshots:
        return history.grouped_weight_snapshots[section_name]
    elif section_name in history.segment_weight_snapshots:
        return history.segment_weight_snapshots[section_name]
    return []

def visualize_sectioned_trajectory(history, dataset_name, section_name, color_metric="epoch"):
    snaps = _get_section_snaps(history, section_name)
    coords, pca = _pca_3d(snaps)
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    metric_map = {"epoch":history.epochs, "loss":history.train_losses,
                  "val_acc":history.val_accs, "grad_norm":history.grad_norms}
    cvals = metric_map.get(color_metric, history.epochs)
    ctitle = {"epoch":"Epoch","loss":"Train Loss","val_acc":"Val Acc",
              "grad_norm":"Grad Norm"}[color_metric]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    img = ax.scatter(x, y, z, c=cvals, cmap='viridis', s=40, label='Trajectory')
    ax.plot(x, y, z, color='gray', alpha=0.5, lw=1.5)
    fig.colorbar(img, ax=ax, label=ctitle)
    
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, marker='D', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, marker='D', label='End')
    
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(f"{section_name} Trajectory ({ctitle})")
    ax.legend()
    return fig

def visualize_sectioned_loss_landscape(model, history, X, y, dataset_name, section_name, binary, grid_size=20, subsample=200):
    snaps = _get_section_snaps(history, section_name)
    if len(snaps) < 3: return None
    
    w_final_segment = snaps[-1]
    w_final = torch.tensor(w_final_segment, dtype=torch.float32, device=DEVICE)
    
    S = np.stack(snaps)
    pca = PCA(n_components=2).fit(S)
    d1 = torch.tensor(pca.components_[0], dtype=torch.float32, device=DEVICE)
    d2 = torch.tensor(pca.components_[1], dtype=torch.float32, device=DEVICE)
    
    traj_2d = pca.transform(S)
    traj_a, traj_b = traj_2d[:, 0], traj_2d[:, 1]
    
    scale1 = np.sqrt(pca.explained_variance_[0]) * 1.5
    scale2 = np.sqrt(pca.explained_variance_[1]) * 1.5
    alphas = np.linspace(-scale1, scale1, grid_size)
    betas  = np.linspace(-scale2, scale2, grid_size)
    
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(subsample, len(X)), replace=False)
    Xs = X[idx].to(DEVICE)
    ys = y[idx].to(DEVICE)
    criterion = nn.BCELoss() if binary else nn.CrossEntropyLoss()
    
    model.eval()
    orig_state = copy.deepcopy(model.state_dict())
    
    w_final_full_np = history.weight_snapshots[-1]

    # Dynamic parameter loading logic
    if section_name in history.layer_weight_snapshots:
        layer_module = model.get_submodule(section_name)
        def load_params(flat_tensor):
            params = list(layer_module.parameters())
            idx_p = 0
            for p in params:
                n = p.numel()
                p.data.copy_(flat_tensor[idx_p:idx_p+n].view_as(p).to(p.device))
                idx_p += n
    elif section_name in history.grouped_weight_snapshots:
        gdef = history.group_definitions[section_name]
        def load_params(flat_tensor):
            idx_p = 0
            for ln in gdef["layers"]:
                lm = model.get_submodule(ln)
                if gdef["type"] in ("weights_only", "full"):
                    p = lm.weight
                    n = p.numel()
                    p.data.copy_(flat_tensor[idx_p:idx_p+n].view_as(p).to(p.device))
                    idx_p += n
                if gdef["type"] in ("biases_only", "full"):
                    p = lm.bias
                    n = p.numel()
                    p.data.copy_(flat_tensor[idx_p:idx_p+n].view_as(p).to(p.device))
                    idx_p += n
    elif section_name in history.segment_weight_snapshots:
        seg_def = history.segment_definitions[section_name]
        def load_params(flat_tensor_segment):
            modified_full = w_final_full_np.copy()
            if isinstance(seg_def, tuple):
                start, end = seg_def
                modified_full[start:end] = flat_tensor_segment.cpu().numpy()
            else:
                modified_full[seg_def] = flat_tensor_segment.cpu().numpy()
            model.load_flat_params(torch.tensor(modified_full, dtype=torch.float32, device=DEVICE))
    else:
        return None

    losses = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(tqdm(alphas, desc=f"Sec Landscape {section_name}", disable=True)):
        for j, b in enumerate(betas):
            load_params(w_final + a*d1 + b*d2)
            with torch.no_grad():
                losses[i, j] = criterion(model(Xs), ys).item()
                
    model.load_state_dict(orig_state)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(alphas, betas, losses, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Loss')
    ax.plot(traj_a, traj_b, 'w.-', lw=2, ms=6, label='Trajectory')
    ax.plot(traj_a[0], traj_b[0], 'go', ms=12, label='Start')
    ax.plot(traj_a[-1], traj_b[-1], 'r*', ms=15, label='End')
    ax.set_title(f"{section_name} Isolated Loss Landscape")
    ax.set_xlabel('Direction 1 (PCA)')
    ax.set_ylabel('Direction 2 (PCA)')
    ax.legend()
    fig.tight_layout()
    return fig

def visualize_sectioned_weight_density(history, dataset_name, section_name, n_bins=50):
    snaps = _get_section_snaps(history, section_name)
    all_w = np.concatenate(snaps)
    w_min, w_max = all_w.min(), all_w.max()
    hist = np.zeros((len(snaps), n_bins))
    for i, s in enumerate(snaps):
        hist[i] = np.histogram(s, bins=n_bins, range=(w_min, w_max))[0]
        
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(hist, aspect='auto', cmap='magma',
                   extent=[w_min, w_max, len(snaps), 0], origin='upper')
    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Weight value'); ax.set_ylabel('Snapshot idx')
    ax.set_title(f'{section_name} — Weight Density Evolution')
    fig.tight_layout()
    return fig

def visualize_sectioned_grad_norm(history, dataset_name, section_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if section_name in history.layer_weight_snapshots:
        w_key = f"{section_name}.weight"
        b_key = f"{section_name}.bias"
        w_vals = [d.get(w_key, 0) for d in history.per_layer_grads]
        b_vals = [d.get(b_key, 0) for d in history.per_layer_grads]
        ax.plot(history.epochs, w_vals, lw=2, label=f"{section_name}.weight", color='blue')
        ax.plot(history.epochs, b_vals, lw=2, label=f"{section_name}.bias", color='red')
    elif section_name in history.grouped_weight_snapshots:
        gdef = history.group_definitions[section_name]
        combined_grads = []
        for d in history.per_layer_grads:
            total_sq = 0.0
            for ln in gdef["layers"]:
                if gdef["type"] in ("weights_only", "full"):
                    total_sq += d.get(f"{ln}.weight", 0)**2
                if gdef["type"] in ("biases_only", "full"):
                    total_sq += d.get(f"{ln}.bias", 0)**2
            combined_grads.append(np.sqrt(total_sq))
        ax.plot(history.epochs, combined_grads, lw=2, label=f"Combined Grad Norm", color='purple')
    elif section_name in history.segment_weight_snapshots:
        vals = history.segment_grad_norms.get(section_name, [])
        if vals:
            ax.plot(history.epochs, vals, lw=2, label=f"{section_name} Grad Norm", color='purple')
        else:
            ax.text(0.5, 0.5, "Gradient data not available.", ha='center', va='center', transform=ax.transAxes)
    else:
        return None
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm (L2)")
    ax.set_title(f"{section_name} — Gradient Norm Evolution")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# ==================================================================
# 7. PySide6 Worker & GUI
# ==================================================================
class StdoutRedirector:
    def __init__(self, signal):
        self.signal = signal
    def write(self, text):
        if text.strip():
            self.signal.emit(text)
    def flush(self):
        pass

class TrainingWorker(QThread):
    plot_ready = Signal(str, object)
    animation_ready = Signal(str, object, object)
    sectioned_ready = Signal(dict)  # Emits payload dict
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, dataset_name, cfg, hidden_layers, use_pca_dirs):
        super().__init__()
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.hidden_layers = hidden_layers
        self.use_pca_dirs = use_pca_dirs
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.log_signal)
        try:
            set_global_seed()
            self.log_signal.emit(f"Device: {DEVICE} | vmap: {_HAS_VMAP}\n")
            
            if self.dataset_name == "moons":
                X_tr, X_te, y_tr, y_te = prepare_moons_data()
                self.run_dataset("moons", self.cfg, X_tr, X_te, y_tr, y_te, 2, 1, True)
            elif self.dataset_name == "digits":
                X_tr, X_te, y_tr, y_te = prepare_digits_data()
                self.run_dataset("digits", self.cfg, X_tr, X_te, y_tr, y_te, 64, 10, False)
                
            self.log_signal.emit("Training Complete!\n")
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"\n[ERROR] Training failed:\n{tb}\n")
        finally:
            sys.stdout = old_stdout
            self.finished_signal.emit()

    def run_dataset(self, name, cfg, X_tr, X_te, y_tr, y_te, in_dim, out_dim, binary):
        self.log_signal.emit(f"\n{'='*60}\n{name.upper()}  ({in_dim}→{out_dim})\n{'='*60}\n")
        model = MLP(in_dim, self.hidden_layers, out_dim, dropout=0.0, binary=binary)
        coll = setup_vector_db(f"{name}_weights")
        h = train_and_vectorize(model, X_tr, y_tr, X_te, y_te, coll, name, cfg, binary, stop_check=lambda: self._stop_flag)

        if self._stop_flag or not h.weight_snapshots:
            self.log_signal.emit(f"Skipping visualization for {name} due to interruption or lack of data.\n")
            return

        fig = visualize_metrics(h, name)
        self.plot_ready.emit(f"{name}: Metrics", fig)

        fig = visualize_layer_shift(h, name)
        self.plot_ready.emit(f"{name}: Layer Shift", fig)
        
        fig = visualize_pca_variance(h, name)
        if fig: self.plot_ready.emit(f"{name}: PCA Variance", fig)
        
        fig = visualize_lr_loss(h, name)
        if fig: self.plot_ready.emit(f"{name}: LR vs Loss", fig)

        if binary:
            fig = visualize_sigmoid_confidence(model, X_te, y_te, name)
            self.plot_ready.emit(f"{name}: Sigmoid Confidence", fig)
            
            fig, ani = animate_decision_boundary(model, h, X_tr, y_tr, name)
            if fig and ani:
                self.animation_ready.emit(f"{name}: Decision Boundary Animation", fig, ani)
        else:
            fig = visualize_receptive_fields(model, name, n_units=16)
            self.plot_ready.emit(f"{name}: Receptive Fields", fig)
            
            fig = visualize_confusion_matrix(model, X_te, y_te, name)
            self.plot_ready.emit(f"{name}: Confusion Matrix", fig)

        sectioned_payload = {
            "dataset_name": name,
            "history": h,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()}, 
            "in_dim": in_dim, 
            "out_dim": out_dim, 
            "hidden": self.hidden_layers, 
            "binary": binary,
            "X_tr": X_tr,
            "y_tr": y_tr,
            "use_pca_dirs": self.use_pca_dirs
        }
        self.sectioned_ready.emit(sectioned_payload)

        mid = len(h.weight_snapshots)//2
        qvec = h.weight_snapshots[mid].tolist()
        res = coll.query(query_embeddings=[qvec], n_results=5, include=["metadatas","distances"])
        self.log_signal.emit(f"\n[SimpleVectorDB] Nearest snapshots to epoch {h.epochs[mid]}:\n")
        for m, d in zip(res['metadatas'][0], res['distances'][0]):
            self.log_signal.emit(f"  ep={m['epoch']:3d} | val_acc={m['val_acc']:.4f} | cos_dist={d:.4f}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==================================================================
# 8. Sectioned Viewer Widget
# ==================================================================
class SectionedViewer(QWidget):
    """A unified widget to view the Whole Network, Combined Layers, and Individual Layers."""
    def __init__(self, payload: dict):
        super().__init__()
        self.payload = payload
        self.history = payload["history"]
        self.dataset_name = payload["dataset_name"]
        self.use_pca_dirs = payload["use_pca_dirs"]
        
        self.model = MLP(payload["in_dim"], payload["hidden"], payload["out_dim"], binary=payload["binary"])
        self.model.load_state_dict(payload["model_state"])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        self.binary = payload["binary"]
        self.X_tr = payload["X_tr"]
        self.y_tr = payload["y_tr"]
        
        # Track all matplotlib figures created in this viewer to prevent memory leaks
        self._figures = []
        
        self.layout = QVBoxLayout(self)
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Select View:"))
        self.combo = QComboBox()
        top_layout.addWidget(self.combo, 1)
        self.layout.addLayout(top_layout)
        
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack, 1)
        
        self._build_ui()

    def _build_ui(self):
        self._add_global_section()
        
        combined_fig = visualize_combined_sectioned_trajectory(self.history, self.dataset_name)
        self._add_combined_section(combined_fig)
        
        layer_names = list(self.history.layer_weight_snapshots.keys())
        for ln in layer_names:
            self._add_layer_section(ln)
            
        if hasattr(self.history, 'grouped_weight_snapshots'):
            for group_name in self.history.grouped_weight_snapshots.keys():
                self._add_layer_section(group_name)
                
        if hasattr(self.history, 'segment_weight_snapshots'):
            for seg_name in self.history.segment_weight_snapshots.keys():
                self._add_layer_section(seg_name)
            
        self.combo.currentIndexChanged.connect(self.stack.setCurrentIndex)

    def _create_placeholder_scroll(self, text):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        placeholder = QLabel(text)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 12pt;")
        scroll.setWidget(placeholder)
        return scroll

    def _add_global_section(self):
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        sub_tabs = QTabWidget()
        
        traj_scroll = self._create_placeholder_scroll("Click to load 3D Trajectory...")
        sub_tabs.addTab(traj_scroll, "3D Trajectory")
        
        density_scroll = self._create_placeholder_scroll("Click to load Weight Density...")
        sub_tabs.addTab(density_scroll, "Weight Density")
        
        grad_scroll = self._create_placeholder_scroll("Click to load Gradient Norm...")
        sub_tabs.addTab(grad_scroll, "Gradient Norm")
        
        land_scroll = self._create_placeholder_scroll("Click to compute Global Loss Landscape...")
        sub_tabs.addTab(land_scroll, "Loss Landscape")
        
        sub_tabs.currentChanged.connect(lambda idx, st=sub_tabs: self._lazy_load_global_plot(idx, st))
        v_layout.addWidget(sub_tabs)
        self.combo.addItem("Whole Network (Global)")
        self.stack.addWidget(container)

    def _add_combined_section(self, combined_fig):
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        sub_tabs = QTabWidget()
        
        traj_scroll = QScrollArea()
        traj_scroll.setWidgetResizable(True)
        traj_scroll.setWidget(FigureCanvas(combined_fig))
        self._figures.append(combined_fig)
        sub_tabs.addTab(traj_scroll, "3D Trajectory (All Layers)")
        
        v_layout.addWidget(sub_tabs)
        self.combo.addItem("All Layers Combined")
        self.stack.addWidget(container)

    def _add_layer_section(self, section_name):
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        sub_tabs = QTabWidget()
        
        traj_scroll = self._create_placeholder_scroll("Click to load 3D Trajectory...")
        sub_tabs.addTab(traj_scroll, "3D Trajectory")
        
        density_scroll = self._create_placeholder_scroll("Click to load Weight Density...")
        sub_tabs.addTab(density_scroll, "Weight Density")
        
        grad_scroll = self._create_placeholder_scroll("Click to load Gradient Norm...")
        sub_tabs.addTab(grad_scroll, "Gradient Norm")
        
        land_scroll = self._create_placeholder_scroll(f"Click to compute Isolated Loss Landscape for {section_name}...")
        sub_tabs.addTab(land_scroll, "Isolated Loss Landscape")
        
        sub_tabs.currentChanged.connect(lambda idx, st=sub_tabs, name=section_name: self._lazy_load_sectioned_plot(idx, st, name))
        
        v_layout.addWidget(sub_tabs)
        self.combo.addItem(section_name)
        self.stack.addWidget(container)

    def _lazy_load_global_plot(self, tab_idx, sub_tabs):
        tab_text = sub_tabs.tabText(tab_idx)
        scroll_area = sub_tabs.widget(tab_idx)
        current_widget = scroll_area.widget()
        if isinstance(current_widget, FigureCanvas): return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fig = None
            if tab_text == "3D Trajectory":
                fig = visualize_trajectory(self.history, self.dataset_name, color_metric="epoch")
            elif tab_text == "Weight Density":
                fig = visualize_weight_heatmap(self.history, self.dataset_name)
            elif tab_text == "Gradient Norm":
                fig = visualize_global_grad_norm(self.history, self.dataset_name)
            elif tab_text == "Loss Landscape":
                fig = visualize_loss_landscape(self.model, self.history, self.X_tr, self.y_tr, self.dataset_name, self.binary, grid_size=30, use_pca_directions=self.use_pca_dirs)
            
            if fig:
                old_widget = scroll_area.takeWidget()
                if old_widget is not None:
                    old_widget.deleteLater()
                scroll_area.setWidget(FigureCanvas(fig))
                self._figures.append(fig)
        except Exception as e:
            err_label = QLabel(f"Error computing plot:\n{str(e)}")
            err_label.setWordWrap(True)
            old_widget = scroll_area.takeWidget()
            if old_widget is not None:
                old_widget.deleteLater()
            scroll_area.setWidget(err_label)
        finally:
            QApplication.restoreOverrideCursor()

    def _lazy_load_sectioned_plot(self, tab_idx, sub_tabs, section_name):
        tab_text = sub_tabs.tabText(tab_idx)
        scroll_area = sub_tabs.widget(tab_idx)
        current_widget = scroll_area.widget()
        if isinstance(current_widget, FigureCanvas): return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fig = None
            if tab_text == "3D Trajectory":
                fig = visualize_sectioned_trajectory(self.history, self.dataset_name, section_name)
            elif tab_text == "Weight Density":
                fig = visualize_sectioned_weight_density(self.history, self.dataset_name, section_name)
            elif tab_text == "Gradient Norm":
                fig = visualize_sectioned_grad_norm(self.history, self.dataset_name, section_name)
            elif tab_text == "Isolated Loss Landscape":
                fig = visualize_sectioned_loss_landscape(self.model, self.history, self.X_tr, self.y_tr, self.dataset_name, section_name, self.binary, grid_size=20)
            
            if fig:
                old_widget = scroll_area.takeWidget()
                if old_widget is not None:
                    old_widget.deleteLater()
                scroll_area.setWidget(FigureCanvas(fig))
                self._figures.append(fig)
        except Exception as e:
            err_label = QLabel(f"Error computing plot:\n{str(e)}")
            err_label.setWordWrap(True)
            old_widget = scroll_area.takeWidget()
            if old_widget is not None:
                old_widget.deleteLater()
            scroll_area.setWidget(err_label)
        finally:
            QApplication.restoreOverrideCursor()

    def cleanup_figures(self):
        """Closes all matplotlib figures associated with this viewer to free memory."""
        for fig in self._figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figures.clear()

# ==================================================================
# 9. Main Window
# ==================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Training Visualizer (v14 - Unified Sections)")
        self.resize(1300, 850)
        
        self.active_animations = []
        self.active_figures = []  # Track main window figures
        self.worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        
        form = QFormLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Moons Default", "Digits Default", "Deep Network", "Fast Demo", "High LR"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Moons", "Digits"])
        self.dataset_combo.currentTextChanged.connect(self.update_defaults)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 500)
        self.epochs_spin.setValue(60)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 512)
        self.batch_spin.setValue(32)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        
        self.arch_edit = QLineEdit("16, 8")
        self.arch_edit.setToolTip("Enter hidden layers separated by commas (e.g., 32, 16, 8)")
        
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["PCA (Trajectory)", "Random (Filter-Norm)"])
        
        form.addRow("Presets:", self.preset_combo)
        form.addRow("Dataset:", self.dataset_combo)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch Size:", self.batch_spin)
        form.addRow("Learning Rate:", self.lr_spin)
        form.addRow("Hidden Layers:", self.arch_edit)
        form.addRow("Landscape Dir:", self.dir_combo)
        
        self.run_button = QPushButton("▶ Start Training")
        self.run_button.setStyleSheet("font-weight: bold; height: 30px; background-color: #4CAF50; color: white;")
        self.run_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("⏹ Stop Training")
        self.stop_button.setStyleSheet("font-weight: bold; height: 30px; background-color: #f44336; color: white;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)

        self.clear_button = QPushButton("🧹 Clear Tabs")
        self.clear_button.clicked.connect(self.clear_tabs)

        control_layout.addLayout(form)
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addStretch()
        
        right_split = QWidget()
        right_layout = QVBoxLayout(right_split)
        
        self.tab_widget = QTabWidget()
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        
        right_layout.addWidget(self.tab_widget, 7)
        right_layout.addWidget(QLabel("Console Output:"))
        right_layout.addWidget(self.log_console, 2)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_split, 1)

    def apply_preset(self, text):
        if text == "Custom": return
        
        self.dataset_combo.currentTextChanged.disconnect(self.update_defaults)
        
        if text == "Moons Default":
            self.dataset_combo.setCurrentText("Moons")
            self.epochs_spin.setValue(60)
            self.batch_spin.setValue(32)
            self.lr_spin.setValue(0.01)
            self.arch_edit.setText("16, 8")
        elif text == "Digits Default":
            self.dataset_combo.setCurrentText("Digits")
            self.epochs_spin.setValue(60)
            self.batch_spin.setValue(32)
            self.lr_spin.setValue(0.001)
            self.arch_edit.setText("128, 64")
        elif text == "Deep Network":
            self.dataset_combo.setCurrentText("Digits")
            self.epochs_spin.setValue(100)
            self.batch_spin.setValue(64)
            self.lr_spin.setValue(0.001)
            self.arch_edit.setText("256, 128, 64, 32")
        elif text == "Fast Demo":
            self.dataset_combo.setCurrentText("Moons")
            self.epochs_spin.setValue(20)
            self.batch_spin.setValue(64)
            self.lr_spin.setValue(0.05)
            self.arch_edit.setText("8, 4")
        elif text == "High LR":
            self.dataset_combo.setCurrentText("Digits")
            self.epochs_spin.setValue(60)
            self.batch_spin.setValue(128)
            self.lr_spin.setValue(0.01)
            self.arch_edit.setText("64, 32")
            
        self.dataset_combo.currentTextChanged.connect(self.update_defaults)
        self.preset_combo.setCurrentText("Custom")

    def update_defaults(self, text):
        if text == "Moons":
            self.lr_spin.setValue(0.01)
            self.arch_edit.setText("16, 8")
        else:
            self.lr_spin.setValue(0.001)
            self.arch_edit.setText("128, 64")

    def start_training(self):
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_button.setText("Training in progress...")
        self.clear_tabs()
        
        dataset = self.dataset_combo.currentText().lower()
        cfg = TrainConfig(
            epochs=self.epochs_spin.value(), 
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value()
        )
        
        try:
            hidden_str = self.arch_edit.text()
            hidden_layers = [int(h.strip()) for h in hidden_str.split(",") if h.strip()]
            if not hidden_layers:
                raise ValueError("No hidden layers provided")
        except ValueError:
            self.append_log("[ERROR] Invalid Hidden Layers format. Use comma-separated integers (e.g., 16, 8).")
            self.on_training_finished()
            return
            
        use_pca_dirs = (self.dir_combo.currentText().startswith("PCA"))
        
        self.worker = TrainingWorker(dataset, cfg, hidden_layers, use_pca_dirs)
        self.worker.plot_ready.connect(self.display_figure)
        self.worker.animation_ready.connect(self.display_animation)
        self.worker.sectioned_ready.connect(self.display_sectioned)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("\n[GUI] Stop requested. Finishing current epoch...\n")
            self.stop_button.setEnabled(False)

    def on_training_finished(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ Start Training")
        self.stop_button.setEnabled(False)
        self.worker = None

    def append_log(self, text):
        self.log_console.appendPlainText(text)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def display_figure(self, tab_name, fig):
        index = self.find_tab_index(tab_name)
        if index != -1:
            # Clean up old figure to prevent memory leak
            old_widget = self.tab_widget.widget(index)
            if isinstance(old_widget, QScrollArea):
                canvas = old_widget.widget()
                if isinstance(canvas, FigureCanvas) and canvas.figure in self.active_figures:
                    plt.close(canvas.figure)
                    self.active_figures.remove(canvas.figure)
            self.tab_widget.removeTab(index)
            
        canvas = FigureCanvas(fig)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(canvas)
        
        self.active_figures.append(fig)
        index = self.tab_widget.addTab(scroll, tab_name)
        self.tab_widget.setCurrentIndex(index)

    def display_animation(self, tab_name, fig, ani):
        index = self.find_tab_index(tab_name)
        if index != -1:
            old_widget = self.tab_widget.widget(index)
            if isinstance(old_widget, FigureCanvas) and old_widget.figure in self.active_figures:
                plt.close(old_widget.figure)
                self.active_figures.remove(old_widget.figure)
            self.tab_widget.removeTab(index)
            
        canvas = FigureCanvas(fig)
        self.active_figures.append(fig)
        index = self.tab_widget.addTab(canvas, tab_name)
        self.tab_widget.setCurrentIndex(index)
        self.active_animations.append(ani)

    def display_sectioned(self, payload):
        tab_name = f"{payload['dataset_name']}: Sectioned Analysis"
        index = self.find_tab_index(tab_name)
        if index != -1:
            old_widget = self.tab_widget.widget(index)
            if isinstance(old_widget, SectionedViewer):
                old_widget.cleanup_figures()
            self.tab_widget.removeTab(index)
            
        viewer = SectionedViewer(payload)
        # QTabWidget.addWidget does not exist — use addTab(widget, label).
        idx = self.tab_widget.addTab(viewer, tab_name)
        self.tab_widget.setCurrentIndex(idx)

    def find_tab_index(self, name):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == name:
                return i
        return -1

    def clear_tabs(self):
        # Stop and clear animations
        for ani in self.active_animations:
            try:
                ani.event_source.stop()
            except Exception:
                pass
        self.active_animations.clear()
        
        # Clean up global figures
        for fig in self.active_figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        self.active_figures.clear()
        
        # Clean up sectioned viewer figures
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, SectionedViewer):
                widget.cleanup_figures()
                
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)

    def closeEvent(self, event):
        self.clear_tabs()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())