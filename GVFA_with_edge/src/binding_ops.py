"""
Shared hypervector binding for GVFA and auxiliary VSA helpers.

- circular: FFT-based binding (circular convolution in the hypervector domain, HRR-style).
- elementwise: Hadamard product (element-wise multiplication).
"""
import torch
from torch.fft import fft, ifft

_BINDING_MODES = frozenset({"circular", "elementwise"})


def bind_hypervectors(x: torch.Tensor, y: torch.Tensor, mode: str = "circular", dim: int = -1) -> torch.Tensor:
    """
    Bind two hypervector tensors of the same shape.

    Parameters
    ----------
    x, y : Tensor
        Same shape; binding is applied along ``dim`` (default: last = hypervector dimension).
    mode : {"circular", "elementwise"}
        ``circular`` — FFT multiply / IFFT (circular convolution).
        ``elementwise`` — ``x * y``.
    dim : int
        Dimension along which each vector lives (default -1).
    """
    if mode not in _BINDING_MODES:
        raise ValueError(f"mode must be one of {_BINDING_MODES}, got {mode!r}")
    if mode == "elementwise":
        return x * y
    fft_x = fft(x, dim=dim)
    fft_y = fft(y, dim=dim)
    return torch.real(ifft(torch.mul(fft_x, fft_y), dim=dim))
