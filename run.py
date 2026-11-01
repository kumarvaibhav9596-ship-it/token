"""Batch-run a folder of 16 kHz mono PCM WAV customer clips through the agent."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT if (ROOT / "agent.py").is_file() else ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import voice_adapter  # noqa: E402
from agent import AGENT  # noqa: E402


def safe_stem(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return value[:100] or "voice_message"


def process_folder(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    if not input_dir.is_dir():
        raise ValueError("The input path must be an existing folder.")
    if not voice_adapter.configured():
        raise RuntimeError("GNANI_API_KEY is not set. The batch voice run needs the Gnani key in the environment.")
    accepted_types = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    clips = sorted((path for path in input_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in accepted_types),
                   key=lambda path: path.name.casefold())
    if not clips:
        raise ValueError("No supported audio files were found (WAV, MP3, FLAC, OGG, M4A).")
    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = failed = 0
    used_names: dict[str, int] = {}
    for clip in clips:
        stem = safe_stem(clip.stem)
        used_names[stem.casefold()] = used_names.get(stem.casefold(), 0) + 1
        count = used_names[stem.casefold()]
        basename = f"{stem}_{count}" if count > 1 else stem
        try:
            transcript = voice_adapter.transcribe_file(clip)
            result = AGENT.respond(f"batch-{basename}", transcript)
            reply_audio = voice_adapter.synthesize_stream(result["answer"])
            (output_dir / f"{basename}.txt").write_text(
                f"Transcript: {transcript}\n\nReply: {result['answer']}\n", encoding="utf-8")
            (output_dir / f"{basename}.wav").write_bytes(reply_audio)
            (output_dir / f"{basename}.json").write_text(json.dumps({
                "input_file": clip.name,
                "transcript": transcript,
                "reply": result["answer"],
                "decision": result["decision"],
                "sources": result.get("sources", []),
                "engine": result.get("engine", "deterministic"),
                "tool_calls": result.get("tool_calls", []),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            succeeded += 1
            print(f"Processed {clip.name}")
        except Exception as error:
            # Save a plain, sanitized failure marker; provider diagnostics and keys stay private.
            (output_dir / f"{basename}.error.txt").write_text(
                f"This audio file could not be processed ({type(error).__name__}). Check its WAV format, Gnani access, and run logs.\n",
                encoding="utf-8")
            failed += 1
            print(f"Could not process {clip.name}; see its .error.txt file.")
    return succeeded, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe, answer, and speak every supported audio file in a folder.")
    parser.add_argument("audio_folder", type=Path, help="Folder of WAV, MP3, FLAC, OGG, or M4A clips up to 60 seconds")
    parser.add_argument("output_folder", type=Path, help="Folder for reply text, WAV audio, and cited sources")
    args = parser.parse_args()
    try:
        succeeded, failed = process_folder(args.audio_folder, args.output_folder)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Batch complete: {succeeded} succeeded, {failed} failed. Outputs: {args.output_folder.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
