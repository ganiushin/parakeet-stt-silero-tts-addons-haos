"""Wyoming server for Parakeet TDT 0.6B v3 on Intel NPU via OpenVINO."""
__version__ = "0.1.0"

# Model identifier, and the name of its subdirectory under the data dir.
# Shared with scripts/bootstrap.py so the two cannot drift apart.
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"

__all__ = ["__version__", "MODEL_NAME"]
