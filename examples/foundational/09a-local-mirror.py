#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import tkinter as tk

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport, maybe_capture_participant_camera
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


class MirrorProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
        elif isinstance(frame, InputImageRawFrame):
            await self.push_frame(
                OutputImageRawFrame(image=frame.image, size=frame.size, format=frame.format)
            )
        else:
            await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
}


async def run_bot(transport: BaseTransport):
    logger.info(f"Starting bot")

    tk_root = tk.Tk()
    tk_root.title("Local Mirror")

    tk_transport = TkLocalTransport(
        tk_root,
        TkTransportParams(
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
        ),
    )

    pipeline = Pipeline([transport.input(), MirrorProcessor(), tk_transport.output()])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(),
    )

    async def run_tk():
        while not task.has_finished():
            tk_root.update()
            tk_root.update_idletasks()
            await asyncio.sleep(0.1)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await maybe_capture_participant_camera(transport, client, framerate=30)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await asyncio.gather(runner.run(task), run_tk())


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
