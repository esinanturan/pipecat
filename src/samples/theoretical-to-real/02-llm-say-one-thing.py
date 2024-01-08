import asyncio
import re
from typing import AsyncGenerator

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService

local_joined = False
participant_joined = False

async def main(room_url):
    meeting_duration_minutes = 1
    transport = DailyTransportService(
        room_url,
        None,
        "Say One Thing From an LLM",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    tts = AzureTTSService()
    llm = AzureLLMService()

    messages = [{
        "role": "system",
        "content": "You are an LLM in a WebRTC session, and your text will be converted to audio. Introduce yourself."
    }]
    llm_generator: AsyncGenerator[str, None] = llm.run_llm_async(messages)

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        current_text = ""
        async for text in llm_generator:
            print("text", text)
            current_text += text
            if re.match(r"^.*[.!?]$", text):
                async for audio in tts.run_tts(current_text):
                    transport.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio))
                current_text = ""

    await transport.run()


if __name__ == "__main__":
    asyncio.run(main("https://moishe.daily.co/Lettvins"))