# Simplex Deep LDA

This folder contains a lightweight PyTorch implementation of the Simplex Deep Linear Discriminant
Analysis (Deep LDA) classifier described in our paper: https://arxiv.org/abs/2601.01679

The paper shows that unconstrained maximum-likelihood training of Deep LDA can collapse class
clusters and hurt discrimination. It proposes a constrained LDA head that fixes class means to the
vertices of a regular simplex and enforces spherical covariance. Under these geometric constraints,
maximum-likelihood training is stable and yields well-separated class clusters with competitive
accuracy on image benchmarks.

## What is in this directory

- `lda.py`: PyTorch modules for the simplex-constrained LDA heads.
  - `SimplexLDAHead`: log-likelihood head with fixed simplex means, spherical covariance, and
    trainable class priors.
  - `FisherSimplexLDAHead`: Fisher-criterion head that shares the same geometry and evaluation
    logits as `SimplexLDAHead`.
- `FashionMNIST.ipynb`: Example notebook for training and visualization.
- `CIFAR10_Embeddings.ipynb`: Train fixed identity-covariance Simplex LDA and Softmax
  with matching encoders, then compare their CIFAR-10 test embeddings using PCA.
  Saves `plots/cifar10_lda_vs_softmax_embeddings.png`.
- `data/`, `plots/`: Supporting artifacts.

## Model summary

Given an encoder `z = f_psi(x)` with output dimension `D`, the head fixes `C` class means to a
regular simplex in `R^(C-1)` (optionally zero-padded if `D > C-1`) and uses a spherical covariance
`Sigma = sigma^2 I`. The head learns only the class priors and the variance, while the encoder is
trained end-to-end by maximum likelihood.

Key constraints:
- `D >= C - 1` to embed the simplex.
- The simplex vertices are centered at the origin and rescaled to a fixed pairwise distance.

## Minimal usage

```python
import torch
import torch.nn as nn
from lda import SimplexLDAHead

C = 10
D = C - 1
encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, D),
)
head = SimplexLDAHead(C=C, D=D)

x = torch.randn(32, 1, 28, 28)
z = encoder(x)
logits = head(z)  # log p(y | z) up to a shared constant
loss = nn.CrossEntropyLoss()(logits, torch.randint(0, C, (32,)))
loss.backward()
```

For Fisher-style training, use `FisherSimplexLDAHead` and call `forward(z, y)` to obtain the
negative Fisher ratio; use `logits(z)` at evaluation time.
Pass `covariance_type="identity"` to fix covariance to `I` without covariance parameters;
the default remains learned spherical covariance.

`train_deep_lda_fisher.py` runs the Fisher objective from `lda/CIFAR10_Fisher.ipynb`
with identity covariance on FashionMNIST, CIFAR10, and CIFAR100. It reuses the encoder,
data transforms, seeds, and prior optimization settings from `train_deep_lda_covariances.py`.
Each epoch records Fisher loss and train/test accuracy. Results retain the same JSON
layout, with only the `identity` covariance key.

```bash
cd simplex-lda
python train_deep_lda_fisher.py --epochs 100 --runs 3 --output ./results/deep_lda_fisher.json
```
