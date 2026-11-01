"""Optional Gnani real-time speech-to-text and streamed text-to-speech."""
from __future__ import annotations

import asyncio
import os
import wave
from io import BytesIO
from pathlib import Path


class VoiceUnavailable(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.getenv("GNANI_API_KEY"))


def _pcm_from_wav(audio: bytes) -> bytes:
    try:
        with wave.open(BytesIO(audio), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise ValueError("Voice clip must be 16 kHz, mono, 16-bit PCM WAV.")
            if wav.getnframes() > 16000 * 30:
                raise ValueError("Voice clip must be 30 seconds or shorter.")
            return wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("Upload a valid PCM WAV voice clip.") from error


def transcribe_wav(audio: bytes) -> str:
    if not configured():
        raise VoiceUnavailable("Gnani voice is not configured.")
    if not audio or len(audio) > 2_000_000:
        raise ValueError("Voice clip must be non-empty and 30 seconds or shorter.")
    pcm = _pcm_from_wav(audio)
    try:
        from gnani.stt import GnaniSTTStreamClient, StreamTranscriptEvent
    except ImportError as error:
        raise VoiceUnavailable("The optional Gnani package is not installed.") from error

    async def run() -> str:
        collected: list[str] = []
        async with GnaniSTTStreamClient(api_key=os.environ["GNANI_API_KEY"],
                                        language_code="indic-auto", model="prisma-v3.0",
                                        sample_rate=16000) as stream:
            for offset in range(0, len(pcm), 1024):
                frame = pcm[offset:offset + 1024]
                if len(frame) < 1024:
                    frame += b"\x00" * (1024 - len(frame))
                await stream.send_audio(frame)
                await asyncio.sleep(0.032)
            events = await stream.close()
            for event in events:
                if isinstance(event, StreamTranscriptEvent) and event.text.strip():
                    collected.append(event.text.strip())
        return " ".join(collected).strip()

    transcript = asyncio.run(run())
    if not transcript:
        raise ValueError("Gnani did not return a transcript. Please try speaking again.")
    return transcript


def transcribe_file(audio_path: str | Path) -> str:
    """Transcribe a common input file with Gnani REST for batch submission runs."""
    if not configured():
        raise VoiceUnavailable("Gnani voice is not configured.")
    path = Path(audio_path)
    if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise ValueError("Audio must be WAV, MP3, FLAC, OGG, or M4A.")
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError("Audio file could not be read.") from None
    if size <= 0 or size > 10_000_000:
        raise ValueError("Audio file must be non-empty and no larger than 10 MB.")
    try:
        from gnani.stt import GnaniSTTClient
    except ImportError as error:
        raise VoiceUnavailable("The optional Gnani package is not installed.") from error
    try:
        result = GnaniSTTClient(api_key=os.environ["GNANI_API_KEY"]).transcribe(
            str(path), language_code="indic-auto", model="prisma-v3.0")
    except Exception:
        raise VoiceUnavailable("Gnani could not transcribe this clip. Check access, file format, and clip duration.") from None
    transcript = result.get("transcript") if isinstance(result, dict) else None
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Gnani did not return a transcript. Please try another clip.")
    return transcript.strip()


def synthesize_stream(text: str) -> bytes:
    if not configured():
        raise VoiceUnavailable("Gnani voice is not configured.")
    if not isinstance(text, str) or not text.strip() or len(text) > 2500:
        raise ValueError("Reply text is empty or too long for speech output.")
    try:
        from gnani.tts import AudioConfig, GnaniTTSStreamClient
    except ImportError as error:
        raise VoiceUnavailable("The optional Gnani package is not installed.") from error
    client = GnaniTTSStreamClient(api_key=os.environ["GNANI_API_KEY"])
    return client.synthesize(text, voice="Kaveri", model="timbre-v2.5", language="en-IN",
                             audio_config=AudioConfig(container="wav", sample_rate=16000))
