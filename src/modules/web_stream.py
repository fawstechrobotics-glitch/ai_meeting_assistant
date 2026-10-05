"""
Web Streaming Module for Assistive Real-Time Meeting Tool.

Provides an embedded HTTP MJPEG streaming server with:
1. Live video & HUD overlay streaming
2. Browser webcam capture & upload for Docker on macOS/Windows
3. High-Fidelity Server-Side Speech Audio Endpoint (/api/tts_audio)
   - Compatible with Google Meet "Share tab audio"
   - Plays directly on host Mac speakers via HTML5 Audio
4. Dual-Engine Web Speech Synthesis (SpeechSynthesis API) fallback
5. Auto-unlocking browser audio autoplay policy & animated soundwave HUD
6. Interactive gesture, mode, and subtitle telemetry
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebStreamServer:
    """
    Lightweight, threaded MJPEG streaming server with bidirectional browser webcam
    and audio speech broadcasting (HTML5 Audio + Web Speech API).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        mode_toggle_callback: Optional[Callable[[], str]] = None,
    ) -> None:
        """
        Initializes the web stream server.

        :param host: Bind address (default 0.0.0.0 for Docker container).
        :param port: Listening HTTP port.
        :param mode_toggle_callback: Optional callback to switch ISL/ASL mode.
        """
        self.host = host
        self.port = port
        self.mode_toggle_callback = mode_toggle_callback

        self._current_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Browser webcam upload buffer
        self._incoming_frame: Optional[np.ndarray] = None
        self._incoming_frame_time: float = 0.0
        self._incoming_lock = threading.Lock()

        # Spoken audio queue for browser Web Speech Synthesis / HTML5 audio
        self._speech_counter = 0
        self._latest_speech: Optional[dict[str, Any]] = None
        self._speech_lock = threading.Lock()

        # Audio caching: {phrase_lowercase: (audio_bytes, mime_type)}
        self._audio_cache: dict[str, Tuple[bytes, str]] = {}
        self._audio_cache_lock = threading.Lock()

        # Telemetry cache
        self.telemetry: dict[str, Any] = {
            "mode": "ISL",
            "fps": 0.0,
            "gesture": "None",
            "subtitle": "",
            "source": "Emulated Video",
        }

        # Pre-warm common sign phrases in background
        threading.Thread(target=self._prewarm_audio_cache, daemon=True, name="AudioPrewarm").start()

    def _prewarm_audio_cache(self) -> None:
        """Pre-synthesizes audio for standard sign language vocabulary."""
        common_phrases = [
            "Yes",
            "No",
            "Hello",
            "Good",
            "Thank You",
            "Help",
            "Help Needed",
            "Stop",
            "Meeting",
            "Peace",
            "OK",
            "I Love You",
            "Water",
            "Speech synthesis test successful",
        ]
        for phrase in common_phrases:
            try:
                self.get_audio_data(phrase)
            except Exception as e:
                logger.debug("Failed pre-warming audio for '%s': %s", phrase, e)

    def get_audio_data(self, text: str) -> Tuple[bytes, str]:
        """
        Retrieves or generates synthesized audio bytes (MP3 or WAV) for the phrase.
        Uses gTTS (Google Text-to-Speech) with graceful offline fallback to espeak.
        """
        clean_text = text.strip()
        if not clean_text:
            clean_text = "Yes"
        cache_key = clean_text.lower()

        with self._audio_cache_lock:
            if cache_key in self._audio_cache:
                return self._audio_cache[cache_key]

        # 1. Try gTTS (High quality MP3)
        try:
            from gtts import gTTS  # type: ignore[import-untyped]

            fp = io.BytesIO()
            tts = gTTS(text=clean_text, lang="en", slow=False)
            tts.write_to_fp(fp)
            data = fp.getvalue()
            with self._audio_cache_lock:
                self._audio_cache[cache_key] = (data, "audio/mpeg")
            return data, "audio/mpeg"
        except Exception as gtts_err:
            logger.debug("gTTS generation error for '%s' (%s). Trying espeak...", clean_text, gtts_err)

        # 2. Offline fallback: espeak (WAV output)
        try:
            temp_wav = "/tmp/web_tts.wav"
            subprocess.run(
                ["espeak", "-w", temp_wav, clean_text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            with open(temp_wav, "rb") as f:
                data = f.read()
            with self._audio_cache_lock:
                self._audio_cache[cache_key] = (data, "audio/wav")
            return data, "audio/wav"
        except Exception as espeak_err:
            logger.error("espeak audio generation failed for '%s': %s", clean_text, espeak_err)

        # 3. Minimal silent WAV fallback if all generators fail
        # 44-byte standard empty WAV header
        silent_wav = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
            b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        return silent_wav, "audio/wav"

    def trigger_speech(self, text: str) -> None:
        """
        Publishes a synthesized speech event to connected browser clients.
        """
        with self._speech_lock:
            self._speech_counter += 1
            self._latest_speech = {
                "id": self._speech_counter,
                "text": text,
                "timestamp": time.time(),
            }
            logger.info("[WebStream Audio] Triggered browser speech: '%s' (id=%d)", text, self._speech_counter)

    def get_incoming_frame(self) -> Optional[np.ndarray]:
        """
        Returns the latest frame uploaded from a browser webcam if fresh (< 2.0s old).
        """
        with self._incoming_lock:
            if self._incoming_frame is not None and (time.time() - self._incoming_frame_time) < 2.0:
                self.telemetry["source"] = "Live Browser Webcam"
                return self._incoming_frame.copy()
            self.telemetry["source"] = "Emulated Video"
            return None

    def update_frame(self, frame: np.ndarray, telemetry: Optional[dict[str, Any]] = None) -> None:
        """
        Encodes a BGR frame as JPEG and updates the active stream buffer.
        """
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with self._frame_lock:
                self._current_frame = jpeg.tobytes()
                if telemetry:
                    self.telemetry.update(telemetry)

    def start(self) -> None:
        """Starts the HTTP server thread."""
        server_instance = self

        class StreamHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_HEAD(self) -> None:
                parsed_url = urllib.parse.urlparse(self.path)
                if parsed_url.path == "/api/tts_audio":
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    text = query_params.get("text", ["Yes"])[0]
                    audio_bytes, mime_type = server_instance.get_audio_data(text)
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(audio_bytes)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()

            def do_POST(self) -> None:
                if self.path == "/api/upload_frame":
                    content_length = int(self.headers.get("Content-Length", 0))
                    if content_length > 0:
                        post_body = self.rfile.read(content_length)
                        nparr = np.frombuffer(post_body, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            with server_instance._incoming_lock:
                                server_instance._incoming_frame = img
                                server_instance._incoming_frame_time = time.time()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_error(404, "Unknown API")

            def do_GET(self) -> None:
                parsed_url = urllib.parse.urlparse(self.path)
                clean_path = parsed_url.path

                if clean_path in ("/", "/index", "/index.html"):
                    self._serve_index()
                elif clean_path == "/video_feed":
                    self._serve_mjpeg()
                elif clean_path == "/api/status":
                    self._serve_status()
                elif clean_path == "/api/tts_audio":
                    self._serve_tts_audio(parsed_url.query)
                elif clean_path == "/api/toggle_mode":
                    if server_instance.mode_toggle_callback:
                        new_mode = server_instance.mode_toggle_callback()
                        server_instance.telemetry["mode"] = new_mode
                    self._serve_status()
                elif clean_path.startswith("/api/test_speech"):
                    server_instance.trigger_speech("Speech synthesis test successful")
                    self._serve_status()
                else:
                    self.send_error(404, "Page Not Found")

            def _serve_tts_audio(self, query_str: str) -> None:
                """Serves real audio bytes (MP3/WAV) compatible with browser HTML5 audio & Meet tab share."""
                query_params = urllib.parse.parse_qs(query_str)
                text = query_params.get("text", ["Yes"])[0]
                audio_bytes, mime_type = server_instance.get_audio_data(text)

                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(audio_bytes)

            def _serve_index(self) -> None:
                html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistive Meeting Tool - Real-Time Subtitles & Voice</title>
    <style>
        :root {{
            --bg-color: #0b0f14;
            --card-bg: #151b23;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --border-color: #30363d;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: var(--bg-color); color: var(--text-main); display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 16px; }}
        header {{ width: 100%; max-width: 1120px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }}
        h1 {{ font-size: 1.35rem; color: var(--accent-blue); display: flex; align-items: center; gap: 10px; font-weight: 700; }}
        .badge {{ background: #238636; color: white; padding: 3px 9px; border-radius: 12px; font-size: 0.72rem; font-weight: bold; letter-spacing: 0.5px; }}
        
        .container {{ width: 100%; max-width: 1120px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }}
        .video-box {{ width: 100%; aspect-ratio: 16/9; background: #000; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }}
        .video-box img {{ width: 100%; height: 100%; object-fit: contain; }}
        
        /* Floating HUD Toast Notification */
        #speech-toast {{
            display: none;
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(22, 27, 34, 0.95);
            color: #3fb950;
            padding: 10px 24px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.05rem;
            z-index: 10;
            box-shadow: 0 6px 20px rgba(0,0,0,0.7);
            backdrop-filter: blur(8px);
            border: 2px solid #3fb950;
            display: none;
            align-items: center;
            gap: 12px;
            animation: popIn 0.25s ease-out;
        }}
        @keyframes popIn {{
            from {{ opacity: 0; transform: translate(-50%, 15px) scale(0.95); }}
            to {{ opacity: 1; transform: translate(-50%, 0) scale(1.0); }}
        }}

        /* Soundwave animation bars */
        .sound-wave {{ display: inline-flex; align-items: flex-end; gap: 3px; height: 16px; vertical-align: middle; }}
        .bar {{ width: 3px; height: 4px; background: #3fb950; border-radius: 2px; animation: bounce 0.6s ease-in-out infinite alternate; }}
        .bar:nth-child(2) {{ animation-delay: 0.15s; }}
        .bar:nth-child(3) {{ animation-delay: 0.3s; }}
        .bar:nth-child(4) {{ animation-delay: 0.45s; }}
        @keyframes bounce {{
            0% {{ height: 4px; }}
            100% {{ height: 16px; }}
        }}

        /* Controls Bar */
        .controls {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-top: 1px solid var(--border-color); flex-wrap: wrap; gap: 14px; background: #11161d; }}
        .btn-group {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
        .status-group {{ display: flex; gap: 18px; color: var(--text-muted); font-size: 0.88rem; flex-wrap: wrap; align-items: center; }}
        .status-group span b {{ color: var(--text-main); }}

        button {{ background: var(--accent-blue); color: #0d1117; font-weight: 700; border: none; padding: 9px 16px; border-radius: 6px; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; display: flex; align-items: center; gap: 6px; }}
        button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
        button.active {{ background: #da3633; color: white; }}
        select, input[type=range] {{ background: #21262d; color: var(--text-main); border: 1px solid var(--border-color); padding: 7px 10px; border-radius: 6px; font-size: 0.82rem; outline: none; }}

        /* Meet Tab Audio Banner */
        .meet-tip {{
            width: 100%;
            max-width: 1120px;
            margin-top: 12px;
            background: rgba(31, 111, 235, 0.12);
            border: 1px solid rgba(88, 166, 255, 0.4);
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 0.85rem;
            color: #c9d1d9;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        .meet-tip b {{ color: var(--accent-blue); }}
    </style>
</head>
<body>
    <header>
        <h1>Assistive Meeting Tool <span class="badge">ONLINE</span></h1>
        <div class="status-group">
            <span>Video Feed: <b id="source-val">{server_instance.telemetry.get("source", "Emulated")}</b></span>
            <span>Mode: <b id="mode-val" style="color:#58a6ff;">{server_instance.telemetry.get("mode", "ISL")}</b></span>
            <span>Audio: <b id="audio-lock-badge" style="color:#3fb950;">🔊 Ready (Click to test)</b></span>
        </div>
    </header>

    <div class="container">
        <div class="video-box" onclick="unlockAudio()">
            <img id="stream-img" src="/video_feed" alt="Live Meeting Stream with HUD Subtitles">
            <div id="speech-toast">
                <div class="sound-wave"><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
                <span id="toast-text">Voiced: "Hello"</span>
            </div>
        </div>

        <div class="controls">
            <div class="status-group">
                <span>Subtitles: <b>Live STT</b></span>
                <label for="audio-engine-select" style="display:flex; align-items:center; gap:6px;">
                    Voice Engine:
                    <select id="audio-engine-select">
                        <option value="server">Server Audio (Google MP3 - Meet Compatible)</option>
                        <option value="browser">Browser Web Speech API</option>
                    </select>
                </label>
                <label style="display:flex; align-items:center; gap:6px;">
                    Vol:
                    <input type="range" id="volume-slider" min="0" max="1" step="0.1" value="1.0" style="width:70px;">
                </label>
            </div>
            <div class="btn-group">
                <button id="cam-btn" onclick="toggleWebcam()" style="background:#238636; color:white;">📷 Enable Real Webcam</button>
                <button id="test-audio-btn" onclick="playSpeech('Speech synthesis test successful')" style="background:#1f6feb; color:white;">🔊 Test Voice</button>
                <button onclick="fetch('/api/toggle_mode').then(() => location.reload())">Toggle ISL / ASL</button>
            </div>
        </div>
    </div>

    <div class="meet-tip">
        <span>💡 <b>For Google Meet & Zoom:</b> In Google Meet, click <b>Present now ➔ Chrome Tab</b>, choose this tab (<b>Assistive Meeting Tool</b>), and ensure <b>"Also share tab audio"</b> is checked! The spoken voice will broadcast directly to everyone in your meeting!</span>
    </div>

    <!-- Hidden video & canvas elements for capturing local browser webcam -->
    <video id="local-video" autoplay playsinline muted style="display:none;"></video>
    <canvas id="local-canvas" style="display:none;"></canvas>

    <script>
        let webcamStream = null;
        let captureInterval = null;
        let isUploading = false;
        let lastSpeechId = 0;
        let audioUnlocked = false;

        // Auto-unlock audio autoplay policy on first user interaction
        function unlockAudio() {{
            if (audioUnlocked) return;
            audioUnlocked = true;

            // 1. Resume AudioContext
            try {{
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (AudioCtx) {{
                    const ctx = new AudioCtx();
                    if (ctx.state === 'suspended') ctx.resume();
                }}
            }} catch(e) {{}}

            // 2. Play inaudible click buffer to unlock HTML5 audio in Chrome/Safari
            try {{
                const dummy = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
                dummy.volume = 0.01;
                dummy.play().catch(() => {{}});
            }} catch(e) {{}}

            // 3. Resume SpeechSynthesis
            if ('speechSynthesis' in window) {{
                try {{
                    window.speechSynthesis.resume();
                    window.speechSynthesis.getVoices();
                }} catch(e) {{}}
            }}

            const badge = document.getElementById("audio-lock-badge");
            if (badge) {{
                badge.textContent = "🔊 Audio Active";
                badge.style.color = "#3fb950";
            }}
        }}

        document.addEventListener("click", unlockAudio, {{ once: false }});
        document.addEventListener("touchstart", unlockAudio, {{ once: false }});

        // Preload voices for Web Speech Synthesis
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.onvoiceschanged = () => {{
                window.speechSynthesis.getVoices();
            }};
        }}

        // Unified Voice Synthesis Function (HTML5 Audio + Web Speech API)
        function playSpeech(text) {{
            unlockAudio();
            const cleanText = (text || "").trim();
            if (!cleanText) return;

            const engine = document.getElementById("audio-engine-select").value;
            const volume = parseFloat(document.getElementById("volume-slider").value || "1.0");

            // Show HUD Toast & animated sound bars
            showToast('Voiced: "' + cleanText + '"');

            if (engine === "server") {{
                // Primary Engine: HTML5 Audio from backend /api/tts_audio
                // Captured directly by Google Meet "Share tab audio"
                const audioUrl = '/api/tts_audio?text=' + encodeURIComponent(cleanText) + '&t=' + Date.now();
                const audio = new Audio(audioUrl);
                audio.volume = volume;
                audio.play().catch(err => {{
                    console.warn("Server audio playback failed, falling back to Web Speech:", err);
                    playWebSpeechFallback(cleanText, volume);
                }});
            }} else {{
                // Secondary Engine: Web Speech API
                playWebSpeechFallback(cleanText, volume);
            }}
        }}

        function playWebSpeechFallback(text, volume) {{
            if (!('speechSynthesis' in window)) return;
            try {{
                window.speechSynthesis.resume();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = volume;
                utterance.lang = 'en-US';

                const voices = window.speechSynthesis.getVoices();
                const engVoice = voices.find(v => v.lang.startsWith("en") && !v.name.includes("whisper"));
                if (engVoice) utterance.voice = engVoice;

                window.speechSynthesis.speak(utterance);
            }} catch(err) {{
                console.error("Web Speech error:", err);
            }}
        }}

        function showToast(message) {{
            const toast = document.getElementById("speech-toast");
            const toastText = document.getElementById("toast-text");
            toastText.textContent = message;
            toast.style.display = "inline-flex";
            clearTimeout(toast._timeout);
            toast._timeout = setTimeout(() => {{
                toast.style.display = "none";
            }}, 2800);
        }}

        async function toggleWebcam() {{
            unlockAudio();
            const btn = document.getElementById("cam-btn");
            const video = document.getElementById("local-video");
            const canvas = document.getElementById("local-canvas");
            const ctx = canvas.getContext("2d");

            if (webcamStream) {{
                clearInterval(captureInterval);
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
                btn.textContent = "📷 Enable Real Webcam";
                btn.style.background = "#238636";
                btn.classList.remove("active");
                return;
            }}

            try {{
                webcamStream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }}
                }});
                video.srcObject = webcamStream;
                btn.textContent = "⏹ Stop Real Webcam";
                btn.style.background = "#da3633";
                btn.classList.add("active");

                captureInterval = setInterval(() => {{
                    if (video.videoWidth > 0 && !isUploading) {{
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        isUploading = true;
                        canvas.toBlob(blob => {{
                            if (blob) {{
                                fetch('/api/upload_frame', {{
                                    method: 'POST',
                                    body: blob
                                }}).finally(() => {{ isUploading = false; }});
                            }} else {{
                                isUploading = false;
                            }}
                        }}, 'image/jpeg', 0.75);
                    }}
                }}, 45); // ~22 FPS webcam upload
            }} catch (err) {{
                alert("Camera permission denied or camera not found: " + err.message);
            }}
        }}

        // Poll telemetry and check for speech events every 150ms
        setInterval(async () => {{
            try {{
                const res = await fetch('/api/status');
                const data = await res.json();
                if (data.source) document.getElementById("source-val").textContent = data.source;
                if (data.mode) document.getElementById("mode-val").textContent = data.mode;

                // Trigger voice output when new gesture is confirmed by the AI backend
                if (data.speech && data.speech.id > lastSpeechId) {{
                    // On very first load, set baseline id without blaring old speech
                    if (lastSpeechId === 0) {{
                        lastSpeechId = data.speech.id;
                    }} else {{
                        lastSpeechId = data.speech.id;
                        playSpeech(data.speech.text);
                    }}
                }}
            }} catch (e) {{}}
        }}, 150);
    </script>
</body>
</html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html.encode("utf-8"))))
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            def _serve_status(self) -> None:
                with server_instance._speech_lock:
                    speech_payload = None
                    if server_instance._latest_speech:
                        # Keep speech payload available for 5 seconds to ensure polling client catches it
                        if (time.time() - server_instance._latest_speech["timestamp"]) < 5.0:
                            speech_payload = server_instance._latest_speech

                payload_dict = dict(server_instance.telemetry)
                payload_dict["speech"] = speech_payload

                payload = json.dumps(payload_dict).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _serve_mjpeg(self) -> None:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                try:
                    while True:
                        with server_instance._frame_lock:
                            frame_data = server_instance._current_frame

                        if frame_data is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame_data)))
                            self.end_headers()
                            self.wfile.write(frame_data)
                            self.wfile.write(b"\r\n")

                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), StreamHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            logger.info("Web Stream Server running at http://%s:%d/", self.host, self.port)
        except Exception as err:
            logger.error("Failed to start WebStreamServer on %s:%d: %s", self.host, self.port, err)

    def stop(self) -> None:
        """Shuts down the web server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None
