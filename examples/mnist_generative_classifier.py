"""
A simple proof-of-concept *generative* classifier on MNIST built with FrEIA.

Instead of a discriminative net, we train a normalizing flow ``f`` that maps a
flattened image ``x`` (784-dim) to a latent ``z`` (784-dim). The latent prior is
a Gaussian mixture with one component per class: class ``y`` is ``N(mu_y, I)``
with learnable means ``mu`` of shape ``[10, 784]`` (the simplified IB-INN idea).

Training (class-conditional NLL) for a labelled sample ``(x, y)``::

    z, log_jac = f(x)
    loss = 0.5 * ||z - mu_y||^2 - log_jac          # (per-dim normalised, mean over batch)

Classification, with equal class priors, reduces to nearest latent mean::

    log p(x | y) = -0.5 ||z - mu_y||^2 + log_jac + const
    pred = argmin_y ||z - mu_y||^2                 # log_jac & const cancel across y

Because the model is a proper generative model, we can also *sample* digits:
draw ``z ~ N(mu_y, I)`` and run the flow in reverse, ``f^{-1}(z)``.

Run::

    python examples/mnist_generative_classifier.py --epochs 5

Expected test accuracy after ~5 epochs: roughly 97-98%.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import FrEIA.framework as Ff
import FrEIA.modules as Fm

N_DIM = 28 * 28
N_CLASSES = 10


def pick_device(requested: str) -> torch.device:
    """Resolve the compute device, auto-picking CUDA -> MPS -> CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subnet_fc(c_in: int, c_out: int) -> nn.Module:
    """Subnetwork used inside each affine coupling block."""
    return nn.Sequential(
        nn.Linear(c_in, 512), nn.ReLU(),
        nn.Linear(512, c_out),
    )


def build_inn(n_blocks: int) -> Ff.SequenceINN:
    """A flat normalizing flow: a stack of AllInOneBlocks over 784 dims."""
    inn = Ff.SequenceINN(N_DIM)
    for _ in range(n_blocks):
        # permute_soft=False (hard permutation): recommended over soft for many
        # feature channels (784 here), where soft init is very slow.
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=False)
    return inn


def dequantize(x: torch.Tensor) -> torch.Tensor:
    """Add uniform noise so the flow isn't fed exact 0/1 pixel values."""
    return (x * 255.0 + torch.rand_like(x)) / 256.0


def class_nll(z: torch.Tensor, log_jac: torch.Tensor, y: torch.Tensor,
              mu: torch.Tensor) -> torch.Tensor:
    """Per-dim-normalised negative log-likelihood under each sample's class Gaussian."""
    mu_y = mu[y]                                   # [batch, N_DIM]
    nll = 0.5 * ((z - mu_y) ** 2).sum(dim=1) - log_jac
    return (nll / N_DIM).mean()


@torch.no_grad()
def evaluate(inn: Ff.SequenceINN, mu: torch.Tensor, loader: DataLoader,
             device: torch.device) -> float:
    """Test accuracy via the nearest-latent-mean rule."""
    inn.eval()
    correct = total = 0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        z, _ = inn(x)                              # log_jac is constant across classes
        dist = ((z[:, None, :] - mu[None, :, :]) ** 2).sum(dim=2)  # [batch, N_CLASSES]
        pred = dist.argmin(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def sample_digits(inn: Ff.SequenceINN, mu: torch.Tensor, device: torch.device,
                  per_class: int, out_path: str) -> None:
    """Generate digits per class by sampling z ~ N(mu_y, I) and inverting the flow."""
    from torchvision.utils import save_image

    inn.eval()
    rows = []
    for y in range(N_CLASSES):
        z = mu[y] + torch.randn(per_class, N_DIM, device=device)
        x_gen, _ = inn(z, rev=True)
        rows.append(x_gen.view(per_class, 1, 28, 28))
    grid = torch.cat(rows, dim=0).clamp(0.0, 1.0).cpu()
    save_image(grid, out_path, nrow=per_class)
    print(f"Saved generated samples to {out_path}")


def _class_centroids(proj, y):
    """Mean 2-D position of each class's embedded points."""
    import numpy as np
    return np.stack([proj[y == c].mean(0) for c in range(N_CLASSES)])


def _project_pca(z: torch.Tensor, mu: torch.Tensor, y):
    """Top-2 PCA of the latent codes; project codes and the learned means alike.

    The linear projection places the learned means faithfully, so the stars here
    are the actual mu_y (the nearest-mean decision references).
    """
    mean = z.mean(0, keepdim=True)
    zc = z - mean
    _, s, v = torch.pca_lowrank(zc, q=2, center=False)
    comps = v[:, :2]
    proj = (zc @ comps).numpy()
    stars = ((mu - mean) @ comps).numpy()
    var = (s[:2] ** 2) / (zc ** 2).sum()
    return (proj, stars, "learned class means",
            f"PC1 ({var[0] * 100:.1f}% var)", f"PC2 ({var[1] * 100:.1f}% var)")


def _project_tsne(z: torch.Tensor, mu: torch.Tensor, y):
    """t-SNE embedding of the latent codes.

    t-SNE is non-parametric and neighbour-preserving, so the learned means (which
    sit near the origin relative to the spread-out data) would all collapse to the
    centre. Instead the stars mark each class's empirical centroid in the
    embedding, which is faithful to this 2-D view.
    """
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, z.size(0) // 100))
    proj = TSNE(n_components=2, init="pca", perplexity=perplexity,
                random_state=0).fit_transform(z.numpy())
    return proj, _class_centroids(proj, y), "class centroids", "t-SNE 1", "t-SNE 2"


@torch.no_grad()
def plot_latent(inn: Ff.SequenceINN, mu: torch.Tensor, loader: DataLoader,
                device: torch.device, out_path: str, max_points: int,
                method: str) -> None:
    """Embed the test set's latent codes to 2-D and scatter them by class.

    The flow warps each class to a unit Gaussian around its learned mean ``mu_y``;
    this plot shows how well those clusters separate. Stars mark the class means.
    ``method='tsne'`` (default) reveals cluster structure best; ``'pca'`` is a
    dependency-free linear fallback but is dominated by the top-variance
    directions, which need not be the class-separating ones.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    inn.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        z, _ = inn(x)
        zs.append(z.cpu())
        ys.append(y)
    z = torch.cat(zs, 0)
    y = torch.cat(ys, 0)

    if max_points and z.size(0) > max_points:
        idx = torch.randperm(z.size(0))[:max_points]
        z, y = z[idx], y[idx]
    y = y.numpy()
    mu_cpu = mu.detach().cpu()

    project = _project_tsne if method == "tsne" else _project_pca
    proj, stars, star_label, xlabel, ylabel = project(z, mu_cpu, y)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for c in range(N_CLASSES):
        m = y == c
        ax.scatter(proj[m, 0], proj[m, 1], s=6, color=cmap(c),
                   alpha=0.4, linewidths=0, label=str(c))
    for c in range(N_CLASSES):
        ax.scatter(stars[c, 0], stars[c, 1], s=240, color=cmap(c),
                   marker="*", edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(str(c), (stars[c, 0], stars[c, 1]), color="black",
                    fontsize=10, fontweight="bold", ha="center", va="center", zorder=6)

    # Robust axis limits: clip to the 1st-99th percentile (+margin) of the data so
    # a handful of far-flung latents can't crush everything into a dot.
    lo, hi = np.percentile(proj, [1, 99], axis=0)
    margin = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
    ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"MNIST test set in INN latent space ({method})\nstars = {star_label}")
    ax.legend(title="digit", markerscale=2, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved latent-space plot to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--max-train", type=int, default=0,
                   help="if >0, cap the number of training images (for a quick run)")
    p.add_argument("--no-sample", action="store_true", help="skip generating samples")
    p.add_argument("--plot-latent", action="store_true",
                   help="save a 2-D scatter of the test set in latent space")
    p.add_argument("--plot-method", choices=["tsne", "pca"], default="tsne",
                   help="2-D embedding for the latent plot (tsne needs scikit-learn)")
    p.add_argument("--plot-points", type=int, default=4000,
                   help="max test points to scatter in the latent plot")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(dequantize)])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    if args.max_train > 0:
        train_set = torch.utils.data.Subset(train_set, range(args.max_train))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    inn = build_inn(args.blocks).to(device)
    # One learnable Gaussian mean per class; small init breaks the inter-class symmetry.
    mu = nn.Parameter(0.1 * torch.randn(N_CLASSES, N_DIM, device=device))

    optimizer = torch.optim.Adam(list(inn.parameters()) + [mu], lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        inn.train()
        running = 0.0
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z, log_jac = inn(x)
            loss = class_nll(z, log_jac, y, mu)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        acc = evaluate(inn, mu, test_loader, device)
        print(f"epoch {epoch:2d} | train loss {running / len(train_loader.dataset):.4f} "
              f"| test acc {acc * 100:.2f}%")

    final_acc = evaluate(inn, mu, test_loader, device)
    print(f"\nFinal test accuracy: {final_acc * 100:.2f}%")

    if not args.no_sample:
        sample_digits(inn, mu, device, per_class=10, out_path="mnist_samples.png")

    if args.plot_latent:
        plot_latent(inn, mu, test_loader, device, out_path="mnist_latent.png",
                    max_points=args.plot_points, method=args.plot_method)


if __name__ == "__main__":
    main()
