#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
import sys

from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.livekit import configure
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    (url, token, room_name) = await configure()

    transport = LiveKitTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            vad_events=True,
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. "
            "Your goal is to demonstrate your capabilities in a succinct way. "
            "Your output will be converted to audio so don't include special characters in your answers. "
            "Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    runner = PipelineRunner()

    task = PipelineTask(
        Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ],
        ),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Register an event handler so we can play the audio when the
    # participant joins.
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        await asyncio.sleep(1)
        await task.queue_frame(
            TextFrame(
                "Hello there! How are you doing today? Would you like to talk about the weather?"
            )
        )

    # Register an event handler to receive data from the participant via text chat
    # in the LiveKit room. This will be used to as transcription frames and
    # interrupt the bot and pass it to llm for processing and
    # then pass back to the participant as audio output.
    @transport.event_handler("on_data_received")
    async def on_data_received(transport, data, participant_id):
        logger.info(f"Received data from participant {participant_id}: {data}")
        # convert data from bytes to string
        json_data = json.loads(data)

        await task.queue_frames(
            [
                BotInterruptionFrame(),
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    user_id=participant_id,
                    timestamp=json_data["timestamp"],
                    text=json_data["message"],
                ),
                UserStoppedSpeakingFrame(),
            ],
        )

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
