# 👂 Meeting Side-Ear

Your silent AI companion for enterprise meetings. Listens to audio playing on your Mac — conference calls, tutorials, demos — detects technical terms from AI, data, banking, and IBM technology domains, and shows instant explanations. Viewable on your phone too.

**100% free. Nothing leaves your Mac.**

---

## Architecture

Meeting Side-Ear uses a **dual-pipeline** approach to deliver both a real-time transcript and term detection cards simultaneously:

```
                    ┌─────────────────────────────────────┐
Mac Audio           │           Audio Capture              │
(BlackHole) ───────▶│     continuous 16kHz mono stream     │
                    └──────────┬──────────────┬────────────┘
                               │              │
                    ┌──────────▼───┐  ┌───────▼──────────┐
  PIPELINE A (fast) │  2s chunks   │  │   8s chunks       │ PIPELINE B (slow)
  Live transcript   │  beam_size=1 │  │   beam_size=5     │ Term detection
                    └──────────┬───┘  └───────┬──────────┘
                               │              │
                    ┌──────────▼───┐  ┌───────▼──────────┐
                    │   Whisper    │  │     Whisper        │
                    │  (fast mode) │  │  (accurate mode)   │
                    └──────────┬───┘  └───────┬──────────┘
                               │              │
                    ┌──────────▼───┐  ┌───────▼──────────┐
                    │  Transcript  │  │      Ollama        │
                    │   panel      │  │  (term analysis)   │
                    │  (~2s delay) │  └───────┬──────────┘
                    └──────────────┘          │
                                    ┌─────────▼──────────┐
                                    │    Term Cards        │
                                    │   (~10s delay)       │
                                    └────────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Browser via SSE    │
                                    │  (Mac or iPhone)    │
                                    └────────────────────┘
```

**Pipeline A** — runs every 2 seconds with Whisper in fast mode (beam_size=1). Sends transcribed text to the live transcript panel almost immediately.

**Pipeline B** — accumulates 8 seconds of audio with 1 second of overlap, transcribes with high-accuracy beam search (beam_size=5), then sends to Ollama for term detection. Term cards appear roughly every 8–12 seconds depending on your Mac's speed.

Both pipelines share the same audio input stream. Pipeline B runs in a dedicated worker thread so it never blocks Pipeline A.

---

## Focus Domains

Meeting Side-Ear only surfaces terms from these enterprise domains:

- **AI & Machine Learning** — LLMs, RAG, agents, watsonx, foundation models, embeddings
- **AI & Data Governance** — model risk, explainability, bias, SR 11-7, OpenPages, data lineage
- **Automation & Orchestration** — RPA, AIOps, MLOps, Cloud Pak, intelligent automation
- **Data & Analytics** — data fabric, data mesh, DataOps, ETL, Db2, InfoSphere, Cognos
- **Cloud & Infrastructure** — hybrid cloud, OpenShift, Kubernetes, IBM Z, IBM Power, Red Hat
- **Networking & Security** — zero trust, QRadar, SIEM, SOAR, IAM, encryption, tokenization
- **Banking & Financial Services** — AML, KYC, ISO 20022, SWIFT, Basel III/IV, DORA, fraud detection
- **Integration & Middleware** — IBM MQ, App Connect, API Connect, event streaming, SOA

Generic words, people's names, and vague business terms are explicitly filtered out.

---

## What runs where

| Task | Tool | Cost |
|---|---|---|
| Audio capture | BlackHole (virtual driver) | Free |
| Live transcript | faster-whisper local, beam=1 | Free |
| Term transcription | faster-whisper local, beam=5 | Free |
| AI term analysis | Ollama + llama3.2 local | Free |
| Dashboard | Flask + SSE | Free |

---

## Setup (one time)

### 1. Install BlackHole

BlackHole lets Python capture your Mac's system audio.

1. Download from: https://existential.audio/blackhole/ (grab the **2ch** version)
2. Install it
3. Open **Audio MIDI Setup** (Spotlight it) → click **+** → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your speakers/headphones
5. Go to **System Settings → Sound → Output** → click the gear icon → **Use This Device For Sound Output**

### 2. Make sure Ollama is running

```bash
ollama serve
ollama pull llama3.2   # if not already pulled
```

### 3. Run

```bash
cd techradar
bash run.sh
```

Open on **Mac**: `http://localhost:5001`
Open on **iPhone**: `http://<your-mac-ip>:5001` (same WiFi)

To find your Mac's current IP:
```bash
ipconfig getifaddr en0
```

---

## Tuning

**Whisper model** — tradeoff between accuracy (especially accents) and speed:
```bash
export WHISPER_MODEL=small    # faster, less accurate
export WHISPER_MODEL=medium   # default, good accent handling
export WHISPER_MODEL=large    # most accurate, slowest
```

**Ollama model:**
```bash
export OLLAMA_MODEL=llama3.2:3b   # default
export OLLAMA_MODEL=mistral       # good alternative
export OLLAMA_MODEL=phi3          # smaller/faster
```

**Pipeline timing** — edit these constants in `app.py`:
```python
PREVIEW_CHUNK_SECS  = 2   # Pipeline A: how often transcript updates
ANALYSIS_CHUNK_SECS = 8   # Pipeline B: how often term cards appear
```

---

## Dashboard layout

**Desktop (3 columns):**
- Left: live term cards feed
- Center: detail panel (click "+ more" on any card)
- Right: controls, stats, live transcript

**iPhone (single column + bottom sheet):**
- Full-width term cards
- Tap "+ more" → detail slides up from bottom
- Fixed toolbar at bottom: Start/Stop/Clear
