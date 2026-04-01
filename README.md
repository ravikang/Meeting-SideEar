# 📡 TechRadar — free / local edition

Real-time tech term detector. Listens to your Mac's audio (calls, tutorials, videos), spots tech jargon, and shows instant summaries — viewable on your phone too.

**100% free. Nothing leaves your Mac.**

---

## What runs where

| Task | Tool | Cost |
|---|---|---|
| Audio capture | BlackHole (virtual driver) | Free |
| Transcription | faster-whisper (local) | Free |
| AI analysis | Ollama + llama3.2 (local) | Free |
| Dashboard | Flask web server | Free |

---

## Setup (one time)

### 1. Install BlackHole

BlackHole lets Python capture your Mac's system audio.

1. Download from: https://existential.audio/blackhole/ (grab the **2ch** version)
2. Install it
3. Go to **System Settings → Sound → Output** → select **BlackHole 2ch**

> **Hear audio AND capture it simultaneously:**
> Open **Audio MIDI Setup** (Spotlight it) → click **+** → **Create Multi-Output Device**
> → check both **BlackHole 2ch** and your speakers/headphones.
> Set that Multi-Output Device as your system output.

---

### 2. Make sure Ollama is running with a model

You already have Ollama installed. Just make sure it's running and has llama3.2:

```bash
ollama serve          # start Ollama if it's not running
ollama pull llama3.2  # download the model (one time, ~2GB)
```

To use a different model:
```bash
export OLLAMA_MODEL=mistral   # or phi3, gemma2, etc.
```

---

### 3. Run

```bash
bash run.sh
```

The script will:
- Create a Python virtual environment
- Install faster-whisper and Flask
- Download the Whisper "small" model (~150MB, first run only)
- Check Ollama is running
- Print your phone URL

---

## Open on your phone

The script prints something like:
```
Phone: http://192.168.1.42:5000
```
Open that in Safari/Chrome on your phone (same WiFi). You'll see cards appear in real time.

---

## Tuning

**Whisper model size** — tradeoff between speed and accuracy:
```bash
export WHISPER_MODEL=tiny    # fastest, least accurate
export WHISPER_MODEL=small   # default, great balance
export WHISPER_MODEL=medium  # more accurate, slower
```

**Ollama model:**
```bash
export OLLAMA_MODEL=llama3.2   # default
export OLLAMA_MODEL=mistral    # good alternative
export OLLAMA_MODEL=phi3       # smaller/faster
```

---

## How it works

```
Mac Audio → BlackHole (loopback) → sounddevice
         → 8-second chunks
         → faster-whisper (local transcription)
         → Ollama / llama3.2 (tech term detection)
         → Flask SSE → Browser (Mac or phone)
```
