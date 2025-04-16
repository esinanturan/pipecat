#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import io
import os
from typing import Dict

import numpy as np
import requests
from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import (
    BaseEndOfTurnAnalyzer,
)


class RemoteSmartTurnAnalyzer(BaseEndOfTurnAnalyzer):
    def __init__(self):
        super().__init__()
        self.remote_smart_turn_url = os.getenv("REMOTE_SMART_TURN_URL")

        if not self.remote_smart_turn_url:
            logger.error("REMOTE_SMART_TURN_URL is not set.")
            raise Exception("REMOTE_SMART_TURN_URL environment variable must be provided.")

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        """Serializes a NumPy array into bytes using np.save."""
        logger.debug("Serializing NumPy array to bytes...")
        buffer = io.BytesIO()
        np.save(buffer, audio_array)  # Saves in npy format
        serialized_bytes = buffer.getvalue()
        logger.debug(f"Serialized size: {len(serialized_bytes)} bytes")
        return serialized_bytes

    def _send_raw_request(self, data_bytes: bytes):
        """Sends the bytes as the raw request body."""
        headers = {"Content-Type": "application/octet-stream"}
        logger.debug(
            f"Sending {len(data_bytes)} bytes as raw body to {self.remote_smart_turn_url}..."
        )
        try:
            response = requests.post(
                self.remote_smart_turn_url, data=data_bytes, headers=headers, timeout=60
            )  # Added timeout

            logger.debug("\n--- Response ---")
            logger.debug(f"Status Code: {response.status_code}")

            # Try to logger.debug JSON if successful, otherwise logger.debug text
            if response.ok:
                try:
                    logger.debug("Response JSON:")
                    logger.debug(response.json())
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    logger.debug("Response Content (non-JSON):")
                    logger.debug(response.text)
            else:
                logger.debug("Response Content (Error):")
                logger.debug(response.text)
                response.raise_for_status()  # Raise an exception for bad status codes

        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to send raw request to Daily Smart Turn: {e}")
            raise Exception("Failed to send raw request to Daily Smart Turn.")

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, any]:
        serialized_array = self._serialize_array(audio_array)
        return self._send_raw_request(serialized_array)
