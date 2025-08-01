#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily room and token configuration utilities.

This module provides helper functions for creating and configuring Daily rooms
and authentication tokens. It handles both command-line argument parsing and
environment variable configuration.

The module supports creating temporary rooms for development or using existing
rooms specified via arguments or environment variables.

Required environment variables:

- DAILY_API_KEY - Daily API key for room/token creation
- DAILY_SAMPLE_ROOM_URL (optional) - Existing room URL to use
- DAILY_SAMPLE_ROOM_TOKEN (optional) - Existing token to use

Example::

    import aiohttp
    from pipecat.runner.daily import configure

    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(session)
        # Use room_url and token with DailyTransport
"""

import argparse
import os
from typing import Optional

import aiohttp

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper


async def configure(aiohttp_session: aiohttp.ClientSession):
    """Configure Daily room URL and token from arguments or environment.

    Args:
        aiohttp_session: HTTP session for making API requests.

    Returns:
        Tuple containing the room URL and authentication token.

    Raises:
        Exception: If room URL or API key are not provided.
    """
    (url, token, _) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(
    aiohttp_session: aiohttp.ClientSession, parser: Optional[argparse.ArgumentParser] = None
):
    """Configure Daily room with command-line argument parsing.

    Args:
        aiohttp_session: HTTP session for making API requests.
        parser: Optional argument parser. If None, creates a default one.

    Returns:
        Tuple containing room URL, authentication token, and parsed arguments.

    Raises:
        Exception: If room URL or API key are not provided via arguments or environment.
    """
    if not parser:
        parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=False,
        help="Daily API Key (needed to create an owner token for the room)",
    )

    args, unknown = parser.parse_known_args()

    url = args.url or os.getenv("DAILY_SAMPLE_ROOM_URL")
    key = args.apikey or os.getenv("DAILY_API_KEY")

    if not url:
        raise Exception(
            "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    if not key:
        raise Exception(
            "No Daily API key specified. use the -k/--apikey option from the command line, or set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    # Create a meeting token for the given room with an expiration 2 hours in
    # the future.
    expiry_time: float = 2 * 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)

    return (url, token, args)
