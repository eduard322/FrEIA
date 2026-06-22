# MNIST Generative Classifier with FrEIA — Design

**Date:** 2026-06-22
**Status:** Approved (lightweight PoC)

## Goal

A simple, runnable proof-of-concept that uses the FrEIA framework (invertible
neural networks / normalizing flows) to classify MNIST digits — in the
framework-native *generative* style rather than a plain discriminative net.

## Concept

A normalizing flow `f` maps a flattened MNIST image `x ∈ ℝ⁷⁸⁴` to a latent
`z ∈ ℝ⁷⁸⁴`. The latent prior is a Gaussian mixture with **one component per
class**: class `y` is `N(μ_y, I)`, with learnable means `μ` of shape `[10, 784]`.
This is the simplified IB-INN ("Generative Classifiers as a Basis for
Trustworthy Image Classification") idea.

- **Training (class-conditional NLL).** For `(x, y)`:
  `z, log_jac = f(x)`; `loss = 0.5·‖z − μ_y‖² − log_jac`, mean over batch,
  per-dimension normalized. Backprop trains both the flow and the means `μ`.
- **Classification.** `log p(x|y) = −0.5‖z − μ_y‖² + log_jac + const`. With
  equal class priors, `log_jac` and constants cancel across classes, so the
  prediction is `argmin_y ‖z − μ_y‖²` (nearest latent mean).
- **Bonus.** Generation: sample `z ~ N(μ_y, I)`, run `f⁻¹(z)` to synthesize a
  digit of class `y`, demonstrating the model is genuinely generative.

## Architecture (single script: `examples/mnist_generative_classifier.py`)

1. `subnet_fc(c_in, c_out)` — 2-layer MLP (hidden width 512, ReLU).
2. `build_inn()` — `Ff.SequenceINN(784)` with 8 `Fm.AllInOneBlock`s
   (`subnet_constructor=subnet_fc`, `permute_soft=True`).
3. Learnable class means `μ` as an `nn.Parameter([10, 784])`, included in the
   optimizer.
4. Data: torchvision MNIST → `./data`, flatten to 784, light uniform
   dequantization (add `U(0,1)/256` noise so the flow isn't fed exact pixel
   values), batch size 256.
5. Train ~5 epochs, Adam; auto device (CUDA → MPS → CPU); print loss/epoch.
6. Eval: test accuracy via the nearest-mean rule; print final accuracy
   (expect ~97–98%).
7. Optional: save a grid of generated digits to a PNG.

Each piece (build, loss, train step, eval, sample) is a small standalone
function so it can be read and tested in isolation.

## Environment

Dedicated conda env `freia-poc` (Python 3.11) with `torch`, `torchvision`,
FrEIA `requirements.txt`, and `pip install -e .` for FrEIA itself. (No env on
this machine currently has torch installed.)

## Non-goals (YAGNI)

- No full `GaussianMixtureModel` module (full covariances / conditional feeds) —
  fixed-identity-covariance, learnable-mean mixture is enough for a PoC.
- No convolutional INN / multi-scale architecture — flat 784-dim flow is fine
  for MNIST at PoC scale.
- No hyperparameter tuning, checkpointing, or CLI beyond a couple of flags.

## Expected outcome

A self-contained script that trains in a few minutes (CPU) / faster (MPS) and
reports test accuracy around 97–98%, optionally emitting generated digit
samples.
