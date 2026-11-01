# Running ZentroMart

This project has a clearly separated offline fallback and optional OpenAI or Gemini RAG answer paths. Order verification, retries, refund calculations, and mock ticket tools remain deterministic. Gnani voice is optional and only uses the configured Gnani SDK.

## Start text mode

```powershell
python --version
python -m pip install -r requirements.txt
python .\validate_dataset.py
python -m unittest -v
python .\evaluate_practice.py
python .\agent.py --port 8000
```

Visit <http://127.0.0.1:8000>. Without a configured provider key, the app uses its explicitly labeled offline policy-retrieval fallback.

## Enable Gemini policy RAG

In the same PowerShell window, enter the Gemini key at the hidden prompt:

```powershell
$secureGemini = Read-Host "Gemini API key (hidden)" -AsSecureString
$env:GEMINI_API_KEY = [Net.NetworkCredential]::new("", $secureGemini).Password
Remove-Variable secureGemini
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_MODEL = "gemini-3.8-flash"
$env:GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
```

## Enable OpenAI policy RAG

In a PowerShell window, enter your API key at the hidden prompt. This keeps it out of the command text and project files:

```powershell
$secureOpenAI = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [Net.NetworkCredential]::new("", $secureOpenAI).Password
Remove-Variable secureOpenAI
$env:OPENAI_MODEL = "gpt-4.1"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
python .\agent.py --port 8000
```

Eligible policy questions use OpenAI embeddings and structured Responses API output. A failed provider request falls back to the local retrieval path, which the UI labels as offline. To measure the Step 1 prompting-only baseline against Step 2 RAG on the 52 practice questions, run `python .\evaluate_ai.py --mode both --output .\ai_eval_results.json`; this makes API calls. Both steps use the same question-level score (expected decision plus all required/forbidden rubric facts); RAG evidence hit, exact source coverage, and mean source recall are reported separately. The evaluator saves after each completed step. If a provider error interrupts it, rerun the same command with `--resume` to continue the same model and mode without repeating completed calls. Set API keys only in the current process environment. `.env.example` is a reference and is not automatically loaded.

## Enable Gnani voice

In the same PowerShell window, set your own Gnani key using a hidden prompt, then start the app:

```powershell
$secureGnani = Read-Host "Gnani API key" -AsSecureString
$env:GNANI_API_KEY = [Net.NetworkCredential]::new("", $secureGnani).Password
Remove-Variable secureGnani
python .\agent.py --port 8000
```

Allow microphone access in the browser. The web UI records up to 30 seconds and converts audio to 16 kHz mono PCM before Gnani STT. The answer uses the same support logic and Gnani Kaveri TTS (`timbre-v2.5`, `en-IN`). The web UI and text demo use synthetic customer data only.

## Batch voice run

Put WAV, MP3, FLAC, OGG, or M4A clips (60 seconds and 10 MB maximum per file) in one folder, then run:

```powershell
python .\run.py .\sample_audio .\sample_outputs
```

Each audio file is transcribed through Gnani's file-based REST STT and processed as an independent session. The output folder receives a transcript/reply `.txt`, generated speech `.wav`, and `.json` with the decision, engine label, and cited policy sections. Gnani credentials must be set in the environment first. Failed files receive a sanitized `.error.txt`; the command exits non-zero if any failed.

## Submission guide

See [SUBMISSION_README.md](SUBMISSION_README.md) for architecture, evaluation, limits, and the final submission checklist. The supplied original `README.md` and source data remain unchanged. No public repository, hosted app, live email provider, or live ticket service is configured.
