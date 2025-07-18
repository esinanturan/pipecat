#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local CoreML smart turn analyzer for on-device ML inference.

This module provides a smart turn analyzer that uses CoreML models for
local end-of-turn detection without requiring network connectivity.
"""

from typing import Any, Dict

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

try:
    import coremltools as ct
    import torch
    from transformers import AutoFeatureExtractor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the LocalSmartTurnAnalyzer, you need to `pip install pipecat-ai[local-smart-turn]`."
    )
    raise Exception(f"Missing module: {e}")


class LocalCoreMLSmartTurnAnalyzer(BaseSmartTurn):
    """Local smart turn analyzer using CoreML models.

    Provides end-of-turn detection using locally-stored CoreML models,
    enabling offline operation without network dependencies. Optimized
    for Apple Silicon and other CoreML-compatible hardware.
    """

    def __init__(self, *, smart_turn_model_path: str, **kwargs):
        """Initialize the local CoreML smart turn analyzer.

        Args:
            smart_turn_model_path: Path to directory containing the CoreML model
                and feature extractor files.
            **kwargs: Additional arguments passed to BaseSmartTurn.

        Raises:
            Exception: If smart_turn_model_path is not provided or model loading fails.
        """
        super().__init__(**kwargs)

        if not smart_turn_model_path:
            logger.error("smart_turn_model_path is not set.")
            raise Exception("smart_turn_model_path must be provided.")

        core_ml_model_path = f"{smart_turn_model_path}/coreml/smart_turn_classifier.mlpackage"

        logger.debug("Loading Local Smart Turn model...")
        # Only load the processor, not the torch model
        self._turn_processor = AutoFeatureExtractor.from_pretrained(smart_turn_model_path)
        self._turn_model = ct.models.MLModel(core_ml_model_path)
        logger.debug("Loaded Local Smart Turn")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local CoreML model."""
        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=800,  # Maximum length as specified in training
            return_attention_mask=True,
            return_tensors="pt",
        )

        output = self._turn_model.predict(dict(inputs))
        logits = output["logits"]  # Core ML returns numpy array
        logits_tensor = torch.tensor(logits)
        probabilities = torch.nn.functional.softmax(logits_tensor, dim=1)
        completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)
        prediction = 1 if completion_prob > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": completion_prob,
        }
