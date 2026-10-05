# ==============================================================================
# Assistive Real-Time Meeting Tool - Production Dockerfile
# ==============================================================================
# Multi-stage container configured for OpenCV GUI rendering (X11),
# ALSA / PulseAudio device passthrough, and MediaPipe AI inference.
# ==============================================================================

FROM python:3.10-slim-bookworm AS base

# System environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=20

# Install essential native libraries for OpenCV, MediaPipe, Audio (ALSA/Pulse), and TTS (espeak)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    # OpenCV & OpenGL GUI / X11 rendering dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Audio capture & playback (ALSA, PortAudio, PulseAudio)
    libasound2-dev \
    libasound2-plugins \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    # Text-to-Speech synthesis engines & audio decoders
    espeak-ng \
    libespeak-ng-dev \
    ffmpeg \
    mpg123 \
    # Create espeak alias for pyttsx3 compatibility
    && (which espeak || ln -s /usr/bin/espeak-ng /usr/bin/espeak || true) \
    # Cleanup apt caches
    && rm -rf /var/lib/apt/lists/*

# Create application workspace directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --default-timeout=1000 --retries 20 -r requirements.txt

# Create application user and grant access to audio and video hardware groups
RUN useradd -m -u 1000 appuser && \
    usermod -aG video,audio appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy application source code and configurations
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser .env.example /app/.env.example

# Set user context
USER appuser

# Healthcheck to verify Python runtime environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import cv2, mediapipe, numpy; print('Healthcheck OK')" || exit 1

# Default execution entrypoint
ENTRYPOINT ["python3", "-m", "src.main"]
CMD []
