"""
Quantum simulator device resolution with automatic fallback.

Device preference chain (probed once and cached):

    lightning.gpu    CUDA statevector simulator (when the plugin is
                     installed and a CUDA device is reachable)
    lightning.qubit  C++ multithreaded CPU simulator (fastest portable
                     backend; uses OpenMP for parallel kernel execution)
    default.qubit    reference pure-Python simulator (original backend)

The QAOA and VQC layers resolve their simulator through this module.
If an accelerated backend is unavailable the pipeline falls back to the
reference backend unchanged, so the algorithm and training flow never
change; only the simulator kernel differs.

GPU/CPU data placement: the quantum states live inside the simulator and
never leave it; classical tensors stay on CPU because the GPU plugin is
not installable on this Windows host (and the classical head is tiny), so
no unnecessary tensor/device transfers are introduced.
"""

import os

import pennylane as qml

QAOA_DEVICE_PREFERENCE = os.environ.get("QUANTUM_QAOA_DEVICE", "lightning.gpu")
VQC_DEVICE_PREFERENCE = os.environ.get("QUANTUM_VQC_DEVICE", "lightning.gpu")

_AVAILABLE: dict = {}


def _probe(name: str) -> bool:
    """Return True if the PennyLane device plugin is importable/usable."""
    try:
        qml.device(name, wires=1, shots=None)
        return True
    except Exception:
        return False


def available_devices() -> dict:
    if not _AVAILABLE:
        for name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
            _AVAILABLE[name] = _probe(name)
    return _AVAILABLE


def resolve_device(preference: str = None) -> str:
    """First usable device in the preference chain (probed once, cached)."""
    chain = [preference] if preference else []
    chain += [n for n in ("lightning.gpu", "lightning.qubit", "default.qubit") if n not in chain]
    for name in chain:
        if available_devices().get(name):
            return name
    raise RuntimeError(f"No PennyLane device usable (probed: {chain})")


def qaoa_device(wires):
    return qml.device(resolve_device(QAOA_DEVICE_PREFERENCE), wires=wires, shots=None)


def vqc_device(wires):
    return qml.device(resolve_device(VQC_DEVICE_PREFERENCE), wires=wires, shots=None)