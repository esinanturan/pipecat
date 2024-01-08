import aiohttp
import asyncio
import os

import requests

from collections.abc import AsyncGenerator
from dailyai.services.ai_services import TTSService

class DeepgramTTSService(TTSService):
    def __init__(self, speech_key=None, voice=None):
        super().__init__()

        self.voice = voice or os.getenv("DEEPGRAM_VOICE") or "alpha-asteria-en-v2"
        self.speech_key = speech_key or os.getenv("DEEPGRAM_API_KEY")
    
    def get_mic_sample_rate(self):
        return 24000

    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None]:
        self.logger.info(f"Running deepgram tts for {sentence}")
        base_url = "https://api.beta.deepgram.com/v1/speak"
        request_url = f"{base_url}?model={self.voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self.speech_key}"}
        body = { "text": sentence }
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=body) as r:
                async for data in r.content:
                    yield data