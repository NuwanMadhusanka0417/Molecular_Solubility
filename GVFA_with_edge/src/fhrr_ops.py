"""
Fourier Holographic Reduced Representations (FHRR) helpers.

Hypervectors are complex with one phasor per dimension (|z_i| = 1 after projection).
Binding: element-wise complex multiplication (phase addition).
Bundling: complex superposition then per-dimension projection onto the unit circle.
"""
import math
import torch


def fhrr_to_torus(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project each component onto the unit circle."""
    mag = z.abs().clamp(min=eps)
    return z / mag


def fhrr_bind(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """FHRR binding: element-wise product; renormalize to the torus."""
    out = x * y
    return fhrr_to_torus(out, eps=eps)


def promote_real_to_fhrr(x: torch.Tensor, eps: float = 1e-8, seed: int = 0) -> torch.Tensor:
    """
    Map real-valued features to FHRR phasors.

    Each real value is converted to a phase via a fixed random projection:
    theta = x_real @ w  (one random weight per dim), then z = e^{i*theta}.
    This preserves continuous phase diversity instead of collapsing to {0, pi}.
    """
    if x.is_complex():
        return fhrr_to_torus(x, eps=eps)
    x_f = x.to(torch.float32)
    g = torch.Generator(device=x.device).manual_seed(seed)
    w = torch.randn(1, x_f.shape[-1], generator=g, device=x.device, dtype=torch.float32)
    theta = x_f * w
    return torch.polar(torch.ones_like(theta), theta)


def complex_to_real_stacked(z: torch.Tensor) -> torch.Tensor:
    """Interleave real and imag for sklearn / numpy regression (last dim becomes 2*D)."""
    if not z.is_complex():
        return z
    return torch.view_as_real(z).reshape(*z.shape[:-1], -1)


def complex_spmm(sparse_real: torch.Tensor, dense_complex: torch.Tensor) -> torch.Tensor:
    """
    Sparse (real float32) x Dense (complex64) matrix multiply.

    PyTorch sparse kernels don't support mixed real-sparse x complex-dense,
    so we split into real and imaginary parts, do two real spmm, then recombine.
    """
    if not dense_complex.is_complex():
        return torch.sparse.mm(sparse_real, dense_complex)
    out_r = torch.sparse.mm(sparse_real, dense_complex.real)
    out_i = torch.sparse.mm(sparse_real, dense_complex.imag)
    return torch.complex(out_r, out_i)


def random_phase_matrix(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """
    Canonical FHRR initialization: each element is e^{i*theta} with theta ~ U[0, 2*pi].
    Returns complex64 tensor of shape (rows, cols), all on the unit circle.
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(rows, cols, generator=g) * (2 * math.pi)
    return torch.polar(torch.ones_like(theta), theta)
