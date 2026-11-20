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
  - `SimplexLDAHead`: log-likelihood head with fixed simplex means, fixed covariance
    `I / D` by default, and trainable class priors (`D` is the embedding dimension).
  - `FisherSimplexLDAHead`: Fisher-criterion head that shares the same geometry and evaluation
    logits as `SimplexLDAHead`.
- `FashionMNIST.ipynb`: Example notebook for training and visualization.
- `CIFAR100_SIGReg.ipynb`: CIFAR-100 companion using class-wise `SIGRegNLLLoss`
  with default fixed covariance `I / D` (`D=99`). Uses all 50,000 training
  images, the reference CNN and augmentations, Adam, and a 100-epoch default.
  Includes test NLL/ECE/accuracy and confidence/reliability plots, with no
  holdout for post-hoc temperature fitting. Configure `SIGREG_WEIGHT`,
  `SIGREG_NUM_SLICES`, and `SIGREG_NUM_POINTS` (defaults 1.0, 256, and 65).
  Saves `plots/cifar100_sigregnll_confidence_hist.png` and
  `plots/cifar100_sigregnll_reliability_diagram.png`.
- `FashionMNIST_SIGReg.ipynb`: Fashion-MNIST companion using `SIGRegNLLLoss`
  from `sigreg.py` instead of NLL. Uses `SimplexLDAHead(C, D)` with its default
  fixed covariance `I / D`: `I_9 / 9` for the nine-dimensional embeddings
  (variance `1/9`, standard deviation `1/3`). Adam trains only the encoder
  with the 100-epoch default; simplex means stay fixed and class priors remain
  uniform. SIGReg tests `(z - mu_c) / sigma` against `N(0, I_9)` using the same
  fixed scale as NLL. Includes PCA and confidence/reliability plots.
  Uses all 60,000 training examples, with no calibration holdout or
  temperature scaling; reports predictive NLL/ECE and accuracy on the test set.
  Configure `SIGREG_WEIGHT` (default 1.0; zero recovers NLL),
  `SIGREG_NUM_SLICES` (256), and `SIGREG_SEED` (1729) in setup.
  Saves figures with `sigregnll` filenames under `plots/`.
- `Synthetic_SIGReg.ipynb`: Companion to `Synthetic.ipynb` comparing NLL,
  class-wise covariance matching, and LeJEPA's SIGReg Gaussianity regularizer.
  Uses the same data, encoder, initialization,
  batch order, and epochs 1/5/10/100. `sigreg.py` adapts Algorithm 1 in
  `LeJEPA.pdf` to each class's residuals around its fixed simplex center,
  targeting `N(mu_c, I / D)` using the head's default fixed covariance for all
  three runs (`D=2`, so `Sigma=0.5 I_2`), 256 resampled directions, and 65 training integration
  points on `[-5, 5]` (129 for diagnostics). The finer grids reduce the
  translation aliasing of the paper's 17-point grid at small fixed variance.
  SIGReg uses 5 NLL-only warm-up epochs followed by a 5-epoch linear ramp;
  the NLL and covariance baselines keep their original schedules. The loss
  is generative NLL plus the scheduled `gaussianity_weight`
  times the frequency-weighted class EP statistics (including their `n_c`
  multiplier). Default weight is 1.0; zero recovers NLL. Finite regularization
  encourages Gaussianity rather than guaranteeing it. Overlapping input
  classes also limit an exact match to almost disjoint output Gaussians.
  Covariance matching and
  SIGReg standardize using the variance from `head.cov_diag`; reference circles,
  Gaussian samples, and held-out diagnostics use the same fixed covariance.
  Includes held-out
  characteristic-function diagnostics, a sampled Gaussian reference, Q–Q
  plots, and training curves. Saves `plots/synth_*sigreg*.png` and
  `results/synthetic_sigreg/metrics.json`. Run from this directory with
  PyTorch, NumPy, and Matplotlib; check the regularizer with
  `python -m unittest test_sigreg`.
- `TinyImageNet.ipynb`: Tiny ImageNet counterpart to `CIFAR100.ipynb`, with
  identity-covariance Simplex LDA, held-out temperature calibration, and validation
  confidence/reliability plots. Uses the dataset loaders in `train_tiny_imagenet.py`.
- `TinyImageNet_Softmax.ipynb`: Train the Softmax baseline on extracted Tiny ImageNet
  using the encoder and loaders from `train_tiny_imagenet.py`. Includes validation
  metrics, training curves, and a saved model under `results/tiny_imagenet_softmax/`.
- `CIFAR10_Embeddings.ipynb`: Train fixed identity-covariance Simplex LDA and Softmax
  with matching encoders, then compare their CIFAR-10 test embeddings using PCA.
  Saves `plots/cifar10_lda_vs_softmax_embeddings.png`.
- `CIFAR10_Embeddings_SIGReg.ipynb`: Same CIFAR-10 embedding comparison, replacing
  the LDA model's NLL with `SIGRegNLLLoss` from `sigreg.py` (generative NLL plus
  class-wise SIGReg toward `N(mu_c, I_9)`). Keeps the Softmax cross-entropy
  baseline, matching encoder initialization, augmentations, and 100-epoch
  defaults. Configure `SIGREG_WEIGHT` (default 1.0; zero recovers NLL),
  `SIGREG_NUM_SLICES` (256), and `SIGREG_SEED` (1729) in the setup cell.
  Saves `plots/cifar10_sigregnll_vs_softmax_embeddings.png`.
- `data/`, `plots/`: Supporting artifacts.

## Model summary

Given an encoder `z = f_psi(x)` with output dimension `D`, the head fixes `C` class means to a
regular simplex in `R^(C-1)` (optionally zero-padded if `D > C-1`) and uses a spherical covariance
`Sigma = I / D` by default (`covariance_type="fixed"`). There are no trainable
covariance parameters in this mode: the head learns class priors, while the encoder
is trained end-to-end by maximum likelihood. The default variance is `1 / D`, so
the standard deviation is `1 / sqrt(D)` and the covariance has trace one.

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
head = SimplexLDAHead(C=C, D=D)  # Fixed covariance I / D.

x = torch.randn(32, 1, 28, 28)
z = encoder(x)
logits = head(z)  # log p(y | z) up to a shared constant
loss = nn.CrossEntropyLoss()(logits, torch.randint(0, C, (32,)))
loss.backward()
```

For Fisher-style training, use `FisherSimplexLDAHead` and call `forward(z, y)` to obtain the
negative Fisher ratio; use `logits(z)` at evaluation time.
Pass `covariance_type="identity"` to fix covariance to `I` without covariance parameters;
`covariance_type="fixed"` (the `SimplexLDAHead` default) fixes it to `I / D`.
Use `covariance_type="spherical"` for a trainable shared variance or when loading
an existing learned-spherical checkpoint. The trainable `diag` and `full` modes
also remain available. Fisher and QDA heads retain their own existing defaults.
Check the LDA default with `python -m unittest test_lda_default`.

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

## MM-LDA comparison

`train_mm_lda.py` implements Max-Mahalanobis LDA from Pang, Du, and Zhu,
ICML 2018 (`pang18a.pdf`), Section 3.2 and Algorithm 2. The fixed Gaussian
means satisfy Eq. (4): their squared norms are `--mean-norm-squared`, and
distinct means have inner product `-mean_norm_squared / (num_classes - 1)`.
The construction is the closed form of Algorithm 1, with zero padding for
larger embedding dimensions. Covariance is fixed to identity and class priors
are fixed to uniform for these balanced benchmarks. Only the encoder is learned.

Training minimizes cross entropy of the Gaussian posterior in Eq. (5), using
logits `z @ mu.T - ||mu||^2 / 2 + log(prior)`. The shared `-||z||^2 / 2` cancels
in the posterior. Features are not normalized. The loss is conditional cross
entropy, rather than the baseline's joint Gaussian negative log-likelihood.
Inference uses the same fixed classifier and takes the largest posterior.

The default squared mean norm is 100, as selected in the paper's Section 4.2.
This is distinct from class count, ArcFace's scale, and the existing Simplex
LDA pairwise-distance setting: pairwise distance here is
`sqrt(2 * num_classes * mean_norm_squared / (num_classes - 1))`.

The encoder, transforms, dimensions (9/9/99), Adam settings, batch sizes,
epochs, and run seeds match `train_arcface.py`. These are comparisons under
the local CNN protocol; they do not reproduce the paper's ResNet-32 experiments,
cross-validation, or adversarial evaluation.

```bash
cd simplex-lda
python train_mm_lda.py --epochs 100 --runs 3 --output ./results/mm_lda.json
python train_mm_lda.py --datasets CIFAR100 --mean-norm-squared 100 --output ./results/mm_lda_cifar100.json
python -m unittest test_mm_lda
```

Results use `results[dataset]["mm_lda"]` with the same run and summary layout
as ArcFace, including per-epoch cross-entropy loss, train/test accuracy, and
final-epoch means and population standard deviations. Metadata records the
fixed geometry, priors, covariance, optimizer, and encoder. The JSON is saved
after every completed run.

## PrototypeDL comparison

`train_prototype_dl.py` implements the prototype network from Li et al.,
*Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network
that Explains Its Predictions*, AAAI 2018 (`Li et al. (AAAI 2018).pdf`),
Eqs. (1)-(7), checked against the authors' [PrototypeDL implementation](https://github.com/OscarcarLi/PrototypeDL).
Learned latent prototypes produce squared Euclidean distances. A learned,
bias-free linear layer maps these distances to class logits. Prototypes have
no fixed class assignments, and classification uses all weighted distances.
Prototypes and classifier weights are initialized uniformly in `[0, 1]`,
as in the authors' code; the encoder output remains the baseline's linear
projection with no additional sigmoid or feature normalization.

All components are optimized jointly with
`cross_entropy + lambda_reconstruction * R + lambda_1 * R1 + lambda_2 * R2`.
`R` sums squared pixel errors per image and averages over the batch. `R1`
averages each prototype's squared distance to its closest example in the
current minibatch, as in the paper's approximation; `R2` averages each
example's squared distance to its closest prototype. The three default
weights are `0.05` (paper Section 3.2). The released code's weights
`20, 1, 1, 1` give the same relative weighting, with a total loss 20 times larger.

The script imports the exact encoder, loaders, evaluation, and seed helper from
`train_deep_lda_covariances.py`. Dimensions remain 9/9/99 for
FashionMNIST/CIFAR10/CIFAR100. Adam's default learning rate (`0.001`),
100 epochs, three runs, batch sizes 256/1024, augmentations, and run seeds
1338/1339/1340 match the local baseline. `--lr 0.0001` selects the paper's
Section 3.2 learning rate. The default is 15 prototypes for each dataset,
matching the paper's digit experiment; `--num-prototypes` can change this
independently of class count (for example, 150 on CIFAR100).

To decode the baseline's vector bottleneck, a linear expansion to a
256-channel 7x7 or 8x8 map feeds convolutions and two transposed convolutions,
ending in a sigmoid at 28x28 or 32x32. Reconstruction uses the same augmented
image as the encoder, with CIFAR normalization reversed only for the target,
so the loss is measured in `[0, 1]` pixel space. This is a comparison under the
local CNN protocol, with an adapted decoder, rather than a reproduction of
the paper's 40-dimensional MNIST autoencoder and elastic-deformation experiments.

```bash
cd simplex-lda
python train_prototype_dl.py --epochs 100 --runs 3 --output ./results/prototype_dl.json
python train_prototype_dl.py --datasets CIFAR100 --num-prototypes 150 --output ./results/prototype_dl_cifar100.json
python -m unittest test_prototype_dl
```

Results use `results[dataset]["prototype_dl"]`, with each epoch's four
unweighted loss components, weighted total loss, and train/test accuracy.
Each run records its seed and final accuracies; `summary` reports final-epoch
test means and population standard deviations. Metadata records the model,
objective, and experiment settings. Results are saved after each completed
run; as with the covariance baseline, this script saves metrics, not model
checkpoints. No epoch or hyperparameter selection uses test accuracy.
