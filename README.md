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
- `TinyImageNet.ipynb`: Tiny ImageNet counterpart to `CIFAR100.ipynb`, with
  identity-covariance Simplex LDA, held-out temperature calibration, and validation
  confidence/reliability plots. Uses the dataset loaders in `train_tiny_imagenet.py`.
- `TinyImageNet_Softmax.ipynb`: Train the Softmax baseline on extracted Tiny ImageNet
  using the encoder and loaders from `train_tiny_imagenet.py`. Includes validation
  metrics, training curves, and a saved model under `results/tiny_imagenet_softmax/`.
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

## CIFAR-100 simplex scale sensitivity

`CIFAR100_SimplexScaleSweep.py` sweeps only Simplex LDA over the fixed pairwise
distance `s` between class means. It reuses the encoder, data transforms, Adam
optimizer, and NLL objective from `CIFAR100_LatentDimSweep.py`, with identity
covariance and a fixed latent dimension (default `D=99`). `SimplexLDAHead` now
accepts `s=...`; its default remains `6.0`.

```bash
cd simplex-lda
python3 CIFAR100_SimplexScaleSweep.py --scales 1 2 4 6 8 10 12 --latent-dim 99 --epochs 100 --runs 3
```

Each scale uses the same run seeds (`--seed + run index`). The default output
files are `plots/cifar100_simplex_scale_sweep.json` and
`plots/cifar100_simplex_scale_sweep.png`. The JSON retains each run's training
and test accuracy curves and the mean/population standard deviation of final
test accuracy; the plot shows final accuracy with standard deviation error bars.
Use `--output-dir` to keep separate configurations in separate directories.

## ArcFace comparison

`train_arcface.py` implements ArcFace from Deng et al., CVPR 2019, Eq. (3) and
Algorithm 1 in the attached paper. Features and learned class weights are
L2-normalized inside the head, with no classifier bias. The target cosine is
replaced by `cos(acos(cosine) + m)` and all logits are multiplied by `s` before
cross entropy. Target cosines are clamped just inside `[-1, 1]` for finite acos
gradients; no easy-margin or piecewise monotonicity approximation is used.
Defaults are the paper's `m=0.5` radians and `s=64`. This `s` is an ArcFace logit
scale, distinct from the simplex pairwise-distance parameter.

For the comparison, the script imports the exact `Encoder`, dataset loaders,
evaluation function, and seed helper from `train_deep_lda_covariances.py`.
FashionMNIST/CIFAR10 use `D=9`; CIFAR100 uses `D=99`. The baseline's augmentations,
batch sizes (256/1024), Adam settings, 100 epochs, and three run seeds
(1338, 1339, 1340) are retained. Both encoder and ArcFace class weights are
trained. Train/test classification accuracy uses margin-free cosine scores;
the target margin is used only for training loss. These are classification
experiments with the existing CNN, rather than the paper's face-verification
protocol, ResNet, 512-dimensional embedding, or SGD schedule.

```bash
cd simplex-lda
python train_arcface.py --epochs 100 --runs 3 --output ./results/arcface.json
# Optional: select datasets or adjust the ArcFace hyperparameters.
python train_arcface.py --datasets CIFAR100 --margin 0.5 --scale 64 --output ./results/arcface_cifar100.json
python -m unittest test_arcface
```

The JSON follows the baseline's `results[dataset][method]` layout with method
`arcface`. Each run stores its seed, per-epoch margin loss and train/test accuracy,
and final train/test accuracy. `summary` includes final test means and population
standard deviations. Metadata records the architecture, dimensions, optimizer,
ArcFace settings, and evaluation rule for reporting the experiment. Results are
saved after every completed run. Summary statistics use the final epoch, with
no selection of epochs or hyperparameters by test accuracy.
