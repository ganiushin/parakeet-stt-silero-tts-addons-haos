"""Explicit construction of the onnx-asr Parakeet pipeline around the NPU shims.

``onnx_asr.load_model()`` builds a pipeline for CPU inference: it opens ORT
sessions for the INT8 encoder and decoder/joint (670 MB of model files) and
one per supported input rate for the resampler (seven more, each with its own
thread pool). Every one of them is dead weight here — inference runs on the
NPU through the OpenVINO shims, and Assist always streams 16 kHz.

So rather than loading those sessions and throwing them away, this module
assembles the same ``NemoConformerTdt`` with the shims supplied up front. The
INT8 model files are then never referenced at all, which is what lets the
bootstrap step stop downloading them.

This deliberately reaches into onnx-asr internals; the dependency is pinned to
an exact version for exactly that reason. Unlike a monkey-patched session
factory, everything here fails loudly — at import or at construction — if
upstream moves.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import _AsrWithDecoding
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
from onnx_asr.preprocessors.resampler import Resampler

_LOGGER = logging.getLogger(__name__)


class _ShimmedParakeetTdt(NemoConformerTdt):
    """Parakeet TDT whose encoder and decoder/joint are OpenVINO shims.

    Our direct base's ``__init__`` does exactly three things: initialise the
    decoding layer, open the encoder ORT session, and open the decoder/joint
    ORT session. We want the first and neither of the ONNX graphs, so we call
    the decoding layer's ``__init__`` directly and plug the shims in.

    Only ``config.json`` and ``vocab.txt`` are read; the mel preprocessor is
    the NumPy one, which carries its own filterbanks inside the onnx-asr
    package (``load_model`` picks it too whenever the provider list is plain
    CPU, so this is not a behaviour change).
    """

    def __init__(self, model_dir: Path, encoder: Any, decoder_joint: Any) -> None:
        _AsrWithDecoding.__init__(
            self,
            {"config": model_dir / "config.json", "vocab": model_dir / "vocab.txt"},
            NemoPreprocessorNumpy,
            {},
        )
        self._encoder = encoder
        self._decoder_joint = decoder_joint


class _LazyResampler(Resampler):
    """Resampler that opens its ORT sessions only if one is ever needed.

    ``Resampler.__init__`` eagerly builds an ``InferenceSession`` for every
    supported input rate — seven of them for a 16 kHz target, each with its
    own thread pool sized to the whole CPU. Assist streams 16 kHz, so in
    practice none of them is ever used.
    """

    def __init__(self, sample_rate: int) -> None:
        self._target_sample_rate = sample_rate
        self._preprocessors: dict | None = None

    def __call__(
        self,
        waveforms: np.ndarray,
        waveforms_lens: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sample_rate != self._target_sample_rate and self._preprocessors is None:
            _LOGGER.info(
                "Client sent %d Hz audio; building resamplers to %d Hz",
                sample_rate, self._target_sample_rate,
            )
            Resampler.__init__(self, self._target_sample_rate, {})
        return super().__call__(waveforms, waveforms_lens, sample_rate)


def build(model_dir: str, encoder: Any, decoder_joint: Any) -> TextResultsAsrAdapter:
    """Assemble the Parakeet pipeline around already-constructed shims."""
    asr = _ShimmedParakeetTdt(Path(model_dir), encoder, decoder_joint)
    _LOGGER.info(
        "Pipeline ready: %s features, vocab %d tokens, blank id %d",
        asr._preprocessor_name, asr._vocab_size, asr._blank_idx,
    )
    return TextResultsAsrAdapter(asr, _LazyResampler(asr._get_sample_rate()))
