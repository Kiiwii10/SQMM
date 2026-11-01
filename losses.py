from __future__ import annotations

import torch

DELTA = 1e-8
DIVERGENCES = ("kl", "squared")


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def spectral_energy(signal: torch.Tensor, grid_shape) -> torch.Tensor:
    """Energy per frequency bin, summed over the batch and the feature channels.

    signal: [B, N, K, F] (K signals of F channels, N grid points) -> [K, bins].
    """
    if len(grid_shape) == 1:
        coeffs = torch.fft.rfft(signal, dim=1, norm="ortho")
        return coeffs.abs().pow(2).sum(dim=(0, 3)).transpose(0, 1)
    batch, _, k, f = signal.shape
    grid = signal.permute(0, 2, 3, 1).contiguous().reshape(batch, k, f, *grid_shape)
    dims = tuple(range(grid.dim() - len(grid_shape), grid.dim()))
    coeffs = torch.fft.fftn(grid, dim=dims, norm="ortho")
    return coeffs.abs().pow(2).sum(dim=(0, 2)).reshape(k, -1)


def normalize_energy(energy: torch.Tensor, delta: float = DELTA) -> torch.Tensor:
    """N_delta: a strictly positive distribution over the frequency bins."""
    energy = energy.clamp_min(0.0)
    return (energy + delta) / (energy.sum(dim=-1, keepdim=True) + delta * energy.shape[-1])


def pauli_basis(memory_qubits: int, *, dtype) -> torch.Tensor:
    """All 4^m Pauli words on m qubits, [4^m, 2^m, 2^m] (identity first)."""
    single = [
        torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
        torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
    ]
    words = [torch.ones((1, 1), dtype=torch.complex64)]
    for _ in range(memory_qubits):
        words = [torch.kron(word, pauli) for word in words for pauli in single]
    return torch.stack(words).to(dtype)


def align_loss(band_features: torch.Tensor, memory_states: torch.Tensor, grid_shape,
               divergence: str = "squared") -> torch.Tensor:
    """L_align over the boundaries of the active prefix.

    band_features: [B, N, active, F] real; memory_states: [B, N, active, 2^m, 2^m] complex.
    The memory after the last active band has no reader and is not aligned.
    """
    if divergence not in DIVERGENCES:
        raise ValueError(f"divergence must be one of {DIVERGENCES}")
    boundaries = band_features.shape[-2] - 1
    memory_qubits = (memory_states.shape[-1] - 1).bit_length()
    words = pauli_basis(memory_qubits, dtype=memory_states.dtype)
    pauli_signals = torch.einsum("...ab,wba->...w", memory_states[..., :-1, :, :], words).real
    pauli_signals = pauli_signals.to(band_features.dtype)

    target = normalize_energy(spectral_energy(band_features, grid_shape).cumsum(dim=0)[:boundaries]).detach()
    memory = normalize_energy(spectral_energy(pauli_signals, grid_shape))
    if divergence == "kl":
        tiny = 1e-30
        terms = target * (torch.log(target.clamp_min(tiny)) - torch.log(memory.clamp_min(tiny)))
    else:
        terms = (memory - target).square()
    return terms.mean()
