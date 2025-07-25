/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Pipecat Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebRTC (via Daily).
 * It handles audio/video streaming and manages the connection lifecycle.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 * - The server must implement the /connect endpoint that returns Daily.co room credentials
 * - Browser with WebRTC support
 */

import { LogLevel, PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

/**
 * ChatbotClient handles the connection and media management for a real-time
 * voice and video interaction with an AI bot.
 */
class ChatbotClient {
  constructor() {
    // Initialize client state
    this.pcClient = null;
    this.setupDOMElements();
    this.setupEventListeners();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  setupDOMElements() {
    // Get references to UI control elements
    this.connectBtn = document.getElementById('connect-btn');
    this.disconnectBtn = document.getElementById('disconnect-btn');
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.searchResultContainer = document.getElementById(
      'search-result-container'
    );

    // Create an audio element for bot's voice output
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  setupEventListeners() {
    this.connectBtn.addEventListener('click', () => this.connect());
    this.disconnectBtn.addEventListener('click', () => this.disconnect());
  }

  /**
   * Add a timestamped message to the debug log
   */
  log(message) {
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;

    // Add styling based on message type
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    }

    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Update the connection status display
   */
  updateStatus(status) {
    this.statusSpan.textContent = status;
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.pcClient) return;

    // Get current tracks from the client
    const tracks = this.pcClient.tracks();

    // Set up any available bot tracks
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.pcClient) return;

    // Listen for new tracks starting
    this.pcClient.on(RTVIEvent.TrackStarted, (track, participant) => {
      // Only handle non-local (bot) tracks
      if (!participant?.local && track.kind === 'audio') {
        this.setupAudioTrack(track);
      }
    });

    // Listen for tracks stopping
    this.pcClient.on(RTVIEvent.TrackStopped, (track, participant) => {
      this.log(
        `Track stopped event: ${track.kind} from ${
          participant?.name || 'unknown'
        }`
      );
    });
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  setupAudioTrack(track) {
    this.log('Setting up audio track');
    // Check if we're already playing this track
    if (this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    // Create a new MediaStream with the track and set it as the audio source
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the Pipecat client, initializes devices, and establishes the connection
   */
  async connect() {
    try {
      // Initialize the Pipecat client with a Daily WebRTC transport and our configuration
      this.pcClient = new PipecatClient({
        transport: new DailyTransport(),
        enableMic: true, // Enable microphone for user input
        enableCam: false,
        callbacks: {
          // Handle connection state changes
          onConnected: () => {
            this.updateStatus('Connected');
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = false;
            this.log('Client connected');
          },
          onDisconnected: () => {
            this.updateStatus('Disconnected');
            this.connectBtn.disabled = false;
            this.disconnectBtn.disabled = true;
            this.log('Client disconnected');
          },
          // Handle transport state changes
          onTransportStateChanged: (state) => {
            this.updateStatus(`Transport: ${state}`);
            this.log(`Transport state changed: ${state}`);
            if (state === 'ready') {
              this.setupMediaTracks();
            }
          },
          // Handle search response events
          onBotLlmSearchResponse: this.handleSearchResponse.bind(this),
          // Handle bot connection events
          onBotConnected: (participant) => {
            this.log(`Bot connected: ${JSON.stringify(participant)}`);
          },
          onBotDisconnected: (participant) => {
            this.log(`Bot disconnected: ${JSON.stringify(participant)}`);
          },
          onBotReady: (data) => {
            this.log(`Bot ready: ${JSON.stringify(data)}`);
            this.setupMediaTracks();
          },
          // Transcript events
          onUserTranscript: (data) => {
            // Only log final transcripts
            if (data.final) {
              this.log(`User: ${data.text}`);
            }
          },
          onBotTranscript: (data) => {
            this.log(`Bot: ${data.text}`);
          },
          // Error handling
          onMessageError: (error) => {
            console.log('Message error:', error);
          },
          onError: (error) => {
            console.log('Error:', error);
          },
        },
      });

      //this.pcClient.setLogLevel(LogLevel.DEBUG)

      // Set up listeners for media track events
      this.setupTrackListeners();

      // Initialize audio devices
      this.log('Initializing devices...');
      await this.pcClient.initDevices();

      // Connect to the bot
      this.log('Connecting to bot...');
      await this.pcClient.connect({
        // The baseURL and endpoint of your bot server that the client will connect to
        endpoint: 'http://localhost:7860/connect',
      });

      this.log('Connection complete');
    } catch (error) {
      // Handle any errors during connection
      this.log(`Error connecting: ${error.message}`);
      this.log(`Error stack: ${error.stack}`);
      this.updateStatus('Error');

      // Clean up if there's an error
      if (this.pcClient) {
        try {
          await this.pcClient.disconnect();
        } catch (disconnectError) {
          this.log(`Error during disconnect: ${disconnectError.message}`);
        }
      }
    }
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  async disconnect() {
    if (this.pcClient) {
      try {
        // Disconnect the Pipecat client
        await this.pcClient.disconnect();
        this.pcClient = null;

        // Clean up audio
        if (this.botAudio.srcObject) {
          this.botAudio.srcObject.getTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }

        // Clean up video
        this.searchResultContainer.innerHTML = '';
      } catch (error) {
        this.log(`Error disconnecting: ${error.message}`);
      }
    }
  }

  handleSearchResponse(response) {
    console.log('SearchResponseHelper, received message:', response);
    // Clear existing content
    this.searchResultContainer.innerHTML = '';

    // Create a container for all content
    const contentContainer = document.createElement('div');
    contentContainer.className = 'content-container';

    // Add the search_result
    if (response.search_result) {
      const searchResultDiv = document.createElement('div');
      searchResultDiv.className = 'search-result';
      searchResultDiv.textContent = response.search_result;
      contentContainer.appendChild(searchResultDiv);
    }

    // Add the sources
    if (response.origins) {
      const sourcesDiv = document.createElement('div');
      sourcesDiv.className = 'sources';

      const sourcesTitle = document.createElement('h3');
      sourcesTitle.className = 'sources-title';
      sourcesTitle.textContent = 'Sources:';
      sourcesDiv.appendChild(sourcesTitle);

      response.origins.forEach((origin) => {
        const sourceLink = document.createElement('a');
        sourceLink.className = 'source-link';
        sourceLink.href = origin.site_uri;
        sourceLink.target = '_blank';
        sourceLink.textContent = origin.site_title;
        sourcesDiv.appendChild(sourceLink);
      });

      contentContainer.appendChild(sourcesDiv);
    }

    // Add the rendered_content in an iframe
    if (response.rendered_content) {
      const iframe = document.createElement('iframe');
      iframe.className = 'iframe-container';
      iframe.srcdoc = response.rendered_content;
      contentContainer.appendChild(iframe);
    }

    // Append the content container to the content panel
    this.searchResultContainer.appendChild(contentContainer);
  }
}

// Initialize the client when the page loads
window.addEventListener('DOMContentLoaded', () => {
  new ChatbotClient();
});
