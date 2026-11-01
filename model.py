from __future__ import annotations

import math
import string
from itertools import combinations
from typing import Optional, Sequence

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

ROTATIONS = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
COUPLINGS = {
    "crx": qml.CRX, "cry": qml.CRY, "crz": qml.CRZ,
    "xx": qml.IsingXX, "yy": qml.IsingYY, "zz": qml.IsingZZ, "xy": qml.IsingXY,
}
ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}
PAULIS = {
    "I": torch.eye(2, dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}
MEMORY_MODES = ("read_write", "read", "write", "none")



### Density-matrix simulation ###


_LETTERS = string.ascii_letters[:50]  # 'Z' (index 51) is reserved for the batch axis


def apply_gate(op, rho: torch.Tensor) -> torch.Tensor:
    """rho -> U rho U^dagger for a PennyLane gate acting on some wires of rho.

    ``rho`` has shape [N, 2, ..., 2] with 2n qubit axes (n ket axes, then n bra axes).
    The (possibly batched) gate matrix is contracted onto its own wires only.
    """
    batch, n = rho.shape[0], (rho.dim() - 1) // 2
    wires = [int(w) for w in op.wires]
    k = len(wires)
    u = op.matrix()
    u = (u if torch.is_tensor(u) else torch.tensor(u)).to(device=rho.device, dtype=rho.dtype)
    if u.dim() == 2:
        u = u.unsqueeze(0)
    u = u.reshape((u.shape[0],) + (2,) * (2 * k)).expand((batch,) + (2,) * (2 * k))
    ket, bra = list(_LETTERS[:n]), list(_LETTERS[n:2 * n])
    new_ket, new_bra = list(_LETTERS[2 * n:2 * n + k]), list(_LETTERS[2 * n + k:2 * n + 2 * k])
    out_ket, out_bra = ket.copy(), bra.copy()
    for i, w in enumerate(wires):
        out_ket[w], out_bra[w] = new_ket[i], new_bra[i]
    left = "Z" + "".join(new_ket) + "".join(ket[w] for w in wires)
    right = "Z" + "".join(new_bra) + "".join(bra[w] for w in wires)
    spec = f"{left},Z{''.join(ket)}{''.join(bra)},{right}->Z{''.join(out_ket)}{''.join(out_bra)}"
    return torch.einsum(spec, u, rho, u.conj())


def zero_density_matrix(num_qubits: int) -> torch.Tensor:
    rho = torch.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=torch.complex128)
    rho[0, 0] = 1.0
    return rho


def pauli_word(letters: str, wires: Sequence[int], num_qubits: int) -> torch.Tensor:
    """Matrix of a Pauli word acting on ``wires`` of a ``num_qubits`` register (wire 0 = MSB)."""
    factors = {w: letter for w, letter in zip(wires, letters)}
    matrix = torch.ones((1, 1), dtype=torch.complex128)
    for wire in range(num_qubits):
        matrix = torch.kron(matrix, PAULIS[factors.get(wire, "I")])
    return matrix


def readout_operators(terms: Sequence[str], width: int) -> torch.Tensor:
    """Every term on every wire group of its size: "Z" -> Z_0..Z_{w-1}, "ZZ" -> Z_iZ_j (i<j)."""
    ops = [pauli_word(term, group, width)
           for term in terms for group in combinations(range(width), len(term))]
    return torch.stack(ops)



### Model ###

def default_read_gates(num_bands: int):
    """CRY read in every band; the last band reads with CRY and CRX."""
    return [["cry"]] * (num_bands - 1) + [["cry", "crx"]]


def default_write_gates(num_bands: int):
    """Phase-sensitive XY (partial iSWAP) writes; the last band does not write (no band reads it)."""
    return [["xy"]] * (num_bands - 1) + [[]]


def _cycle_axes(count: int):
    """RY, RX, RY, ... : the upload / memory-initialisation axis of each coordinate / memory qubit."""
    return [("RY", "RX")[i % 2] for i in range(count)]


def _uniform_angles(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape).uniform_(0.0, 2.0 * math.pi))


class Band(nn.Module):
    """Trainable angles of one band: entangling layers, memory reads and writes."""

    def __init__(self, depth: int, width: int, pairs: int, read_gates, write_gates, memory_mode):
        super().__init__()
        self.weights = _uniform_angles(depth, width, 3)
        # Couplings are drawn like every other angle and then set to zero, so the
        # memory lane starts as the identity and is switched on by training.
        if memory_mode in ("read", "read_write"):
            self.read = _uniform_angles(pairs, len(read_gates))
        if memory_mode in ("write", "read_write") and write_gates:
            self.write = _uniform_angles(pairs, len(write_gates))
        with torch.no_grad():
            for name in ("read", "write"):
                if hasattr(self, name):
                    getattr(self, name).zero_()


class SQMM(nn.Module):
    def __init__(
        self,
        *,
        coord_dim: int,
        out_features: int = 1,
        frequencies: dict,
        band_depths: Sequence[int],
        band_widths: Sequence[int],
        memory: Optional[dict] = None,
        upload_axes: Optional[Sequence[str]] = None,
        readout: Sequence[str] = ("Z", "ZZ"),
        readout_mlp: Optional[dict] = None,
    ):
        super().__init__()
        if len(band_widths) != len(band_depths):
            raise ValueError("band_widths and band_depths must have the same length")
        self.coord_dim = int(coord_dim)
        self.band_depths = tuple(int(d) for d in band_depths)
        self.band_widths = tuple(int(w) for w in band_widths)
        self.num_bands = len(self.band_depths)

        memory = dict(memory or {})
        self.memory_mode = memory.get("mode", "read_write")
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory.mode must be one of {MEMORY_MODES}")
        self.memory_qubits = int(memory.get("qubits", 2))
        self.memory_init_axes = list(memory.get("init_axes") or _cycle_axes(self.memory_qubits))
        self.read_gates = [list(g) for g in (memory.get("read_gates") or default_read_gates(self.num_bands))]
        self.write_gates = [list(g) for g in (memory.get("write_gates") or default_write_gates(self.num_bands))]
        self.upload_axes = list(upload_axes or _cycle_axes(self.coord_dim))
        if len(self.memory_init_axes) != self.memory_qubits:
            raise ValueError("memory.init_axes needs one axis per memory qubit")
        if len(self.upload_axes) != self.coord_dim:
            raise ValueError("upload_axes needs one axis per coordinate")
        if len(self.read_gates) != self.num_bands or len(self.write_gates) != self.num_bands:
            raise ValueError("memory.read_gates / write_gates need one gate list per band")

        # Positional encoding PE(x) = [x, sin(w_k x_d), cos(w_k x_d)], w_k = pi 2^k or pi (k+1).
        amount = int(frequencies["amount"])
        if frequencies["mode"] == "dyadic":
            omegas = [math.pi * (2 ** k) for k in range(amount)]
        elif frequencies["mode"] == "consecutive":
            omegas = [math.pi * float(k + 1) for k in range(amount)]
        else:
            raise ValueError("frequencies.mode must be 'dyadic' or 'consecutive'")
        self.register_buffer("omegas", torch.tensor(omegas, dtype=torch.float32))
        self.encoding_dim = self.coord_dim * (1 + 2 * amount)

        # Columns of PE(x) read by the projection of each upload axis. Memory layout of
        # PE(x): [x_0..x_{d-1}, sin(w x_0), cos(w x_0), sin(w x_1), cos(w x_1), ...].
        # 1-D: the single axis reads all of PE(x). 2-D: the RY axis reads
        # [x_0, sin(w x_0), sin(w x_1)] and the RX axis [x_1, cos(w x_0), cos(w x_1)].
        d, k = self.coord_dim, amount
        for axis in range(d):
            cols = [axis] + [d + axis * k + i for i in range(k)] + [d + d * k + axis * k + i for i in range(k)]
            self.register_buffer(f"_axis_columns_{axis}", torch.tensor(cols), persistent=False)

        # Upload angles of band b, data wire q, axis a: (W_{b,a} PE_a(x))_q + c_{b,q}, with one
        # bias vector per band shared by its axes (the same count as a single Linear).
        per_axis = 1 + 2 * amount
        self.upload_projections = nn.ModuleList(
            nn.ModuleList(nn.Linear(per_axis, width, bias=False) for _ in range(d))
            for width in self.band_widths
        )
        bound = 1.0 / math.sqrt(d * per_axis)
        self.upload_biases = nn.ParameterList(
            nn.Parameter(torch.empty(width).uniform_(-bound, bound)) for width in self.band_widths
        )
        # Memory initial angles V PE(x). Without memory the register is still simulated (in
        # |0...0>, never coupled) so the band circuit is exactly the memory model's.
        self.memory_init = (
            nn.Linear(self.encoding_dim, self.memory_qubits) if self.memory_mode != "none" else None
        )

        # Decoder over the concatenated band features.
        self.readout_terms = tuple(readout)
        readouts = [readout_operators(self.readout_terms, w) for w in self.band_widths]
        for b, ops in enumerate(readouts):
            self.register_buffer(f"_readout_{b}", ops, persistent=False)
        self.band_feature_dims = tuple(int(ops.shape[0]) for ops in readouts)
        readout_mlp = dict(readout_mlp or {})
        layers, width_in = [], sum(self.band_feature_dims)
        for width in readout_mlp.get("widths", [5]):
            layers += [nn.Linear(width_in, int(width)), ACTIVATIONS[readout_mlp.get("activation", "gelu")]()]
            width_in = int(width)
        layers.append(nn.Linear(width_in, int(out_features)))
        self.head = nn.Sequential(*layers)

        # Quantum angles. Every data qubit couples to every memory qubit (data-major order).
        self.bands = nn.ModuleList(
            Band(depth, width, width * self.memory_qubits, self.read_gates[b], self.write_gates[b], self.memory_mode)
            for b, (depth, width) in enumerate(zip(self.band_depths, self.band_widths))
        )

    ### inputs ###

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.unsqueeze(-1) * self.omegas
        return torch.cat([x, torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)], dim=-1)

    def upload_angles(self, pe: torch.Tensor, band: int) -> torch.Tensor:
        """[N, width, coord_dim] upload angles of one band."""
        angles = [
            projection(pe.index_select(-1, getattr(self, f"_axis_columns_{axis}"))) + self.upload_biases[band]
            for axis, projection in enumerate(self.upload_projections[band])
        ]
        return torch.stack(angles, dim=-1)

    def initial_memory(self, pe: torch.Tensor) -> torch.Tensor:
        m = self.memory_qubits
        angles = self.memory_init(pe) if self.memory_init is not None else pe.new_zeros(pe.shape[0], m)
        rho = zero_density_matrix(m).to(pe.device).expand(pe.shape[0], -1, -1).reshape((-1,) + (2,) * (2 * m))
        for q, axis in enumerate(self.memory_init_axes):
            rho = apply_gate(ROTATIONS[axis](angles[:, q], wires=q), rho)
        return rho.reshape(-1, 2 ** m, 2 ** m)

    ### one band ###

    def run_band(self, band: int, angles: torch.Tensor, memory: torch.Tensor):
        """Band circuit on |0..0><0..0| (x) memory. Returns (features [N, F], memory [N, D, D])."""
        width, m, block = self.band_widths[band], self.memory_qubits, self.bands[band]
        data = list(range(width))
        mem = list(range(width, width + m))
        pairs = [(q, r) for q in data for r in mem]
        n, dim = memory.shape[0], 2 ** (width + m)

        rho = torch.einsum("ab,ncd->nacbd", zero_density_matrix(width).to(memory), memory)
        rho = rho.reshape((n,) + (2,) * (2 * (width + m)))
        for layer in range(self.band_depths[band]):
            if layer == 0 and hasattr(block, "read"):
                for p, (q, r) in enumerate(pairs):
                    for g, gate in enumerate(self.read_gates[band]):
                        rho = apply_gate(COUPLINGS[gate](block.read[p, g], wires=[r, q]), rho)
            for q in data:
                for a, axis in enumerate(self.upload_axes):
                    rho = apply_gate(ROTATIONS[axis](angles[:, q, a], wires=q), rho)
            for op in qml.StronglyEntanglingLayers(block.weights[layer:layer + 1], wires=data).decomposition():
                rho = apply_gate(op, rho)
        if hasattr(block, "write"):
            for p, (q, r) in enumerate(pairs):
                for g, gate in enumerate(self.write_gates[band]):
                    rho = apply_gate(COUPLINGS[gate](block.write[p, g], wires=[q, r]), rho)

        rho = rho.reshape(n, dim, dim)
        data_rho = qml.math.reduce_dm(rho, indices=data).to(rho.dtype)
        memory = qml.math.reduce_dm(rho, indices=mem).to(rho.dtype)
        ops = getattr(self, f"_readout_{band}").to(data_rho)
        features = torch.diagonal(data_rho.unsqueeze(1) @ ops.unsqueeze(0), dim1=-2, dim2=-1).sum(-1).real
        return features, memory

    ### forward ###-

    def forward(self, coords: torch.Tensor, active_bands: Optional[int] = None, fade: float = 1.0,
                return_memory: bool = False):
        """coords [B, N, coord_dim] -> prediction [B, N, out_features].

        Only the first ``active_bands`` bands run; the decoder sees zeros for the rest and
        the newest band's features scaled by ``fade``. With ``return_memory`` also returns
        the raw band features [B, N, active, F] and the memory state after every active
        band [B, N, active, 2^m, 2^m] (inputs of the alignment loss).
        """
        active = self.num_bands if active_bands is None else int(active_bands)
        prefix = coords.shape[:-1]
        pe = self.encode(coords.reshape(-1, self.coord_dim))
        # Upload angles of ALL bands, as one [N, bands, max width, coord_dim] tensor. A band
        # that has not joined yet is thereby in the graph with an exact-zero gradient, so
        # Adam counts its update steps from the first update on (bias correction included).
        max_width = max(self.band_widths)
        angles = torch.stack([F.pad(self.upload_angles(pe, b), (0, 0, 0, max_width - w))
                              for b, w in enumerate(self.band_widths)], dim=1)
        memory = self.initial_memory(pe)
        features, memories = [], []
        for band in range(active):
            band_features, memory = self.run_band(band, angles[:, band, :self.band_widths[band]], memory)
            features.append(band_features.to(coords.dtype).reshape(*prefix, -1))
            memories.append(memory)

        decoder_inputs = list(features)
        if fade != 1.0:
            decoder_inputs[-1] = decoder_inputs[-1] * fade
        decoder_inputs += [coords.new_zeros(*prefix, f) for f in self.band_feature_dims[active:]]
        prediction = self.head(torch.cat(decoder_inputs, dim=-1))
        if not return_memory:
            return prediction

        width = max(self.band_feature_dims)
        padded = [torch.cat([f, f.new_zeros(*prefix, width - f.shape[-1])], -1) for f in features]
        memory_states = torch.stack(memories, dim=1).reshape(*prefix, active, *memory.shape[-2:])
        return prediction, torch.stack(padded, dim=-2), memory_states

    ### bookkeeping ###

    def parameter_counts(self) -> dict:
        quantum = sum(p.numel() for name, p in self.named_parameters() if name.startswith("bands."))
        total = sum(p.numel() for p in self.parameters())
        return {"trainable_params": total, "quantum_params": quantum, "classical_params": total - quantum}
