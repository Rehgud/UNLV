#!/usr/bin/env python
# coding: utf-8
# HW4 - NN vs CNN for USPS Digit Classification (5-fold stratified CV)
# Dohyeong Ryu

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED         = 42
N_FOLDS      = 5
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
device = torch.device('cpu')   # small models -> CPU is fast and fully reproducible
print('PyTorch', torch.__version__, '| device:', device)

# ---------------------------------------------------------------- data
usps = fetch_openml('usps', version=1)
X, y = usps['data'], usps['target'].astype(int)
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y) - 1                       # labels 1..10 -> digits 0..9
print('X:', X.shape, '| pixel range: [%.1f, %.1f]' % (X.min(), X.max()))
print('y:', y.shape, '| classes:', np.unique(y), '| counts:', np.bincount(y))


# ---------------------------------------------------------------- models
class MLP(nn.Module):
    """Fully-connected NN, identical to HW3 (256-256-128-10)."""
    def __init__(self, n_in=256, n_h1=256, n_h2=128, n_out=10, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_h1), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(n_h1, n_h2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(n_h2, n_out),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class CNN(nn.Module):
    """Two conv blocks (32, 64 filters, 3x3) + 2 max-pools + 2 FC layers."""
    def __init__(self, n_out=10, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),                 # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),                 # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------- training
def run_fold(model_cls, X_tr, y_tr, X_va, y_va, seed, reshape):
    """Train one model on one fold; return per-epoch history + final val accuracy."""
    set_seed(seed)
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    def prep(a):
        t = torch.tensor(a, dtype=torch.float32)
        return t.view(-1, 1, 16, 16) if reshape else t

    train_ds = torch.utils.data.TensorDataset(prep(X_tr),
                                              torch.tensor(y_tr, dtype=torch.long))
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed))

    X_va_t = prep(X_va).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.long).to(device)

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for _ in range(EPOCHS):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += len(xb)

        model.eval()
        with torch.no_grad():
            val_out  = model(X_va_t)
            val_loss = criterion(val_out, y_va_t).item()
            val_acc  = (val_out.argmax(1) == y_va_t).float().mean().item()

        hist['train_loss'].append(loss_sum / total)
        hist['train_acc'].append(correct / total)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

    return hist, hist['val_acc'][-1]


def cross_validate(model_cls, name, reshape):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    histories, accuracies = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])            # fit on training folds only
        X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])
        hist, acc = run_fold(model_cls, X_tr.astype(np.float32), y[tr],
                             X_va.astype(np.float32), y[va],
                             seed=SEED + fold, reshape=reshape)
        histories.append(hist)
        accuracies.append(acc)
        print(f'  [{name}] Experiment {fold}: '
              f'train {len(tr)} / val {len(va)} -> acc {acc:.4f}')
    return histories, accuracies


print('\n=== Training NN (MLP) ===')
nn_hist, nn_acc = cross_validate(MLP, 'NN', reshape=False)
print('\n=== Training CNN ===')
cnn_hist, cnn_acc = cross_validate(CNN, 'CNN', reshape=True)


# ---------------------------------------------------------------- results table
def fmt(accs):
    return [f'{a:.4f}' for a in accs] + [f'{np.mean(accs):.4f} ± {np.std(accs):.4f}']

idx = [f'Experiment {i}' for i in range(1, N_FOLDS + 1)] + ['Average']
results = pd.DataFrame({'NN (Accuracy)': fmt(nn_acc), 'CNN (Accuracy)': fmt(cnn_acc)}, index=idx)
results.index.name = 'USPS 5-fold CV'
results.to_csv('results.csv')
print('\n', results, sep='')


# ---------------------------------------------------------------- sample figure
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for d, ax in enumerate(axes):
    i = np.where(y == d)[0][0]
    ax.imshow(X[i].reshape(16, 16), cmap='gray'); ax.set_title(str(d)); ax.axis('off')
plt.suptitle('USPS sample images (one per class)', y=1.15)
plt.savefig('fig1_samples.png', dpi=150, bbox_inches='tight'); plt.close()


# ---------------------------------------------------------------- learning curves
def plot_curves(histories, name, fname):
    fig, axes = plt.subplots(N_FOLDS, 2, figsize=(11, 3.0 * N_FOLDS))
    ex = np.arange(1, EPOCHS + 1)
    for f, h in enumerate(histories):
        a = axes[f, 0]
        a.plot(ex, h['train_loss'], label='train loss')
        a.plot(ex, h['val_loss'], label='validation loss')
        a.set_title(f'{name} — Experiment {f+1} — Loss')
        a.set_xlabel('epoch'); a.set_ylabel('cross-entropy loss')
        a.legend(); a.grid(alpha=0.3)
        a = axes[f, 1]
        a.plot(ex, h['train_acc'], label='train accuracy')
        a.plot(ex, h['val_acc'], label='validation accuracy')
        a.set_title(f'{name} — Experiment {f+1} — Accuracy')
        a.set_xlabel('epoch'); a.set_ylabel('accuracy')
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=130, bbox_inches='tight'); plt.close()


plot_curves(nn_hist, 'NN', 'fig2_nn_learning_curves.png')
plot_curves(cnn_hist, 'CNN', 'fig3_cnn_learning_curves.png')

# mean validation-accuracy curves, NN vs CNN, on one axis
fig, ax = plt.subplots(figsize=(7, 4.5))
ex = np.arange(1, EPOCHS + 1)
nn_val = np.mean([h['val_acc'] for h in nn_hist], axis=0)
cnn_val = np.mean([h['val_acc'] for h in cnn_hist], axis=0)
ax.plot(ex, nn_val, label='NN mean val acc')
ax.plot(ex, cnn_val, label='CNN mean val acc')
ax.set_xlabel('epoch'); ax.set_ylabel('validation accuracy')
ax.set_title('Mean validation accuracy across 5 folds')
ax.legend(); ax.grid(alpha=0.3)
plt.savefig('fig4_nn_vs_cnn.png', dpi=150, bbox_inches='tight'); plt.close()

print('\nSaved: results.csv, fig1_samples.png, fig2_nn_learning_curves.png,'
      ' fig3_cnn_learning_curves.png, fig4_nn_vs_cnn.png')
