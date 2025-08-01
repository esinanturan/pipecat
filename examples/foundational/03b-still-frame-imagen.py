#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.image import GoogleImageGenService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        video_out_enabled=True,
        video_out_width=1024,
        video_out_height=1024,
    ),
    "webrtc": lambda: TransportParams(
        video_out_enabled=True,
        video_out_width=1024,
        video_out_height=1024,
    ),
}


async def run_bot(transport: BaseTransport):
    logger.info(f"Starting bot")

    imagegen = GoogleImageGenService(
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    task = PipelineTask(
        Pipeline([imagegen, transport.output()]),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frame(TextFrame("a cat in the style of picasso"))
        await task.queue_frame(TextFrame("a dog in the style of picasso"))
        await task.queue_frame(TextFrame("a fish in the style of picasso"))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
