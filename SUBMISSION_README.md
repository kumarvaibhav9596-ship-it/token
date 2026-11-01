# ZentroMart VoiceOps Capstone

An AI customer-support prototype for the fictional ZentroMart retailer. It combines a configurable OpenAI or Gemini prompting-only baseline and policy-grounded RAG with deterministic order verification, refund calculations, and mock human handover. Gnani handles speech input and output. Customer and order records are synthetic.

## Architecture

1. The supplied knowledge base is split at its `[SECTION-ID]` headings. The selected OpenAI or Gemini embedding model embeds current sections and each policy question; cosine similarity selects evidence while preserving source IDs, status, effective dates, and policy precedence. The selected LLM receives sanitized text, recent policy-only context, and retrieved excerpts and returns a structured response. The Step 2 benchmark also adds a matched product catalog row when the message identifies a product; policy documents govern policy rules, while catalog rows supply only listed attributes. Model-declared abstentions are preserved, and citations not present in retrieved evidence are rejected; factual correctness still requires evaluation.
2. The existing Python agent keeps identity checks, one retry, order lookup, business-day calculations, and ticket creation deterministic. Those controls run before the optional language-model answer path. The model cannot call a customer or ticket tool or claim that an action succeeded.
3. Without an OpenAI key, the application stays usable in its clearly labeled offline BM25/rule-based fallback. It does not describe those answers as model-generated.
4. Gnani voice uses the official `gnani-vachana` client for transcription and Kaveri TTS (`timbre-v2.5`, `en-IN`). Voice responses pass through the same support agent. The folder batch command accepts WAV, MP3, FLAC, OGG, or M4A clips up to 60 seconds and saves a transcript/reply `.txt`, generated `.wav`, and `.json` containing decision, response engine, and cited sections for each input clip.

## Included features

- Policy RAG over the supplied current knowledge-base sections, with evidence IDs and an abstain path; the Step 2 benchmark can also pass a matched product catalog row.
- Prompt-only and RAG evaluation command for all 52 supplied practice questions.
- Conversation-aware deterministic order lookup, identity verification, retry limits, refund dates, and human-ticket handover.
- Browser chat and local email/ticket text adapters. Email is a local mock outbox; it never sends externally.
- Optional Gnani speech-to-text and speech playback, plus the audio-folder batch command.
- Dataset validator and offline focused regression suite.

## Requirements and data

- Python 3.10 or later.
- The supplied `Capstone_Project_Dataset_FTTT_2026` extracted beside this README so `data/`, `kb/`, `practice/`, and `tools/` are present. The submission ZIP deliberately does not include a second copy of the dataset.
- An API key for the selected LLM provider. OpenAI and Gemini are supported through `LLM_PROVIDER`; a ChatGPT subscription by itself is not an API credential.
- A Gnani API key and internet access for voice. No speech provider other than Gnani is used.

The submission archive keeps application modules and tests in `src/`, with `run.py`, `.env.example`, the report, and this README at the root. Extract the supplied dataset beside this README so `data/`, `kb/`, `practice/`, and the supplied `tools/` package are present. Install from PowerShell in the extracted project folder:

```powershell
python -m pip install -r requirements.txt
python .\validate_dataset.py
python -m unittest discover -s src -v
python .\src\evaluate_practice.py
```

Set the selected LLM key in the current PowerShell session without putting it in chat or a source file. For Gemini:

```powershell
$secureGemini = Read-Host "Gemini API key (hidden)" -AsSecureString
$env:GEMINI_API_KEY = [Net.NetworkCredential]::new("", $secureGemini).Password
Remove-Variable secureGemini
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_MODEL = "gemini-3.8-flash"
$env:GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
```

For OpenAI, use:

```powershell
$secureOpenAI = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [Net.NetworkCredential]::new("", $secureOpenAI).Password
Remove-Variable secureOpenAI
$env:OPENAI_MODEL = "gpt-4.1"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

$secureGnani = Read-Host "Gnani API key" -AsSecureString
$env:GNANI_API_KEY = [Net.NetworkCredential]::new("", $secureGnani).Password
Remove-Variable secureGnani
```

Start the browser app:

```powershell
python .\src\agent.py --port 8000
```

Open <http://127.0.0.1:8000>. With the selected provider variable set, the health endpoint shows the configured provider and model; without it, the page identifies offline fallback mode. Actual provider access is checked when the first eligible policy question triggers an embedding and response call.

Run the 52-question API-backed baseline and RAG comparison (this makes API calls):

```powershell
python .\src\evaluate_ai.py --mode both --output .\ai_eval_results.json
```

The output contains generated answers, decisions, retrieved IDs, and per-case checks. Step 1 and Step 2 share the same question-level score: expected decision plus required and forbidden rubric facts. RAG evidence overlap, exact gold-source coverage, and mean gold-source recall are separate retrieval measures; Step 1 has no evidence score. If a provider error interrupts the run, keep the same mode/model/output path and resume completed work with:

```powershell
python .\src\evaluate_ai.py --mode both --output .\ai_eval_results.json --resume
```

Only report a score when the JSON status is `complete`. Do not present mock tests or the deterministic offline practice results as live LLM results. For the voice batch command, put WAV, MP3, FLAC, OGG, or M4A clips of at most 60 seconds (10 MB maximum) in one folder:

```powershell
python .\run.py .\sample_audio .\sample_outputs
```

Each input produces a `.txt` transcript and answer, a `.wav` spoken reply, and a `.json` with cited sections. Keep one customer message per file; this batch runner treats files as separate sessions.

## Example interactions

- “Could the pickup agent charge me if I’m sending this back because it didn’t suit me?” → retrieve the relevant current return-fee section, produce a cited answer when supported, or abstain.
- “Where is order 89782837?” → asks for a phone suffix or delivery pincode; no order details are shared before the mock tool verifies it.
- “I need a human agent.” → creates a mock ticket and reports success only if the tool confirms it.
- Voice input follows the same policies and tools after Gnani transcription; audio replies use Gnani TTS.

## Verification and known limits

The offline deterministic suite passes the 52 single questions and 96 conversation turns. The focused regression suite has 65 passing tests in this runtime; two additional live Gnani SDK contract tests are not runnable here because the optional `gnani-vachana` package is not installed. These are not an OpenAI prompting-only-versus-RAG score. The environment used to prepare this package has no LLM API key or `GNANI_API_KEY`, so live model calls, API comparison scores, and a fresh provider-backed voice batch have not been verified. The API code has offline mocked contract tests only. The submission should present the offline results as offline deterministic/RAG results, not as live LLM results.

Other limits: semantic indexing is in-memory and rebuilt on process start; API failures fall back to the separately labeled local retrieval path. The order tool is synthetic and is not real authentication. The app has no live order database, live ticketing system, externally delivered email, or public hosting. The local web server is not accessible to evaluators until code is published or deployed. The final submission still needs a public Git URL and a <=2-minute demo recorded from a successful live run; do not include a dataset copy, `.env`, API keys, or virtual environment.

## Submission checklist

- Publish the source in a public Git repository and add its URL to the submission form.
- Include `.env.example` only; never upload `.env` or any key.
- Keep the report to four pages maximum and replace the explicitly unverified AI comparison with `evaluate_ai.py` results.
- Record a demo no longer than two minutes: a sourced answer with a follow-up, then a human handover or honest “I don't know”. Use your own voice recording.
- Run the audio folder command and include sample text, audio, and source JSON outputs produced by the agent.
- Upload the project report PDF and relevant public code/demo links before the deadline.

