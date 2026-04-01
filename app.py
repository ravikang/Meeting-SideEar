import os
import json
import time
import threading
import queue
import urllib.request
import urllib.error
import re
from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
OLLAMA_URL         = os.environ.get("OLLAMA_URL",    "http://localhost:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL",  "llama3.2")
SAMPLE_RATE        = 16000
CHUNK_DURATION     = 8
OVERLAP            = 1

# ── Load Whisper ──────────────────────────────────────────────
print(f"Loading Whisper '{WHISPER_MODEL_SIZE}'...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("✓ Whisper ready")

# ── Resolve Ollama model name ─────────────────────────────────
def resolve_model(requested):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if requested in names:
            return requested
        matches = [n for n in names if n.startswith(requested)]
        if matches:
            print(f"[Ollama] Using '{matches[0]}' for '{requested}'")
            return matches[0]
        print(f"[Ollama] WARNING: '{requested}' not found. Have: {names}")
    except Exception as e:
        print(f"[Ollama] Could not resolve model: {e}")
    return requested

OLLAMA_MODEL = resolve_model(OLLAMA_MODEL)
print(f"✓ Ollama model: {OLLAMA_MODEL}")

# ── Shared state ──────────────────────────────────────────────
state = {
    "listening":  False,
    "cards":      [],
    "status":     "idle",
}
seen_terms      = set()
sse_clients     = []
audio_thread    = None
card_id_counter = 0

# ── Audio helpers ─────────────────────────────────────────────
def get_blackhole_device():
    for i, d in enumerate(sd.query_devices()):
        if any(k in d["name"].lower() for k in ("blackhole", "loopback", "soundflower")):
            if d["max_input_channels"] > 0:
                return i
    return None

def transcribe(audio: np.ndarray) -> str:
    if not len(audio):
        return ""
    audio = audio.astype(np.float32)
    peak  = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    try:
        segs, _ = whisper_model.transcribe(
            audio, language="en", beam_size=3,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text for s in segs).strip()
    except Exception as e:
        print(f"[Whisper error] {e}")
        return ""

# ── Ollama helpers ────────────────────────────────────────────
def ollama_generate(prompt: str, max_tokens: int = 2000) -> str:
    """Call Ollama and return raw response string."""
    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())["response"].strip()

def extract_json_array(raw: str):
    """Robustly extract a JSON array from a possibly-messy string."""
    # Strip markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()
    # Find the outermost [ ... ]
    start = raw.find("[")
    if start == -1:
        return []
    # Try clean parse
    end = raw.rfind("]")
    if end > start:
        try:
            result = json.loads(raw[start:end+1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Truncated response — extract individual complete objects
    recovered = []
    for m in re.finditer(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw[start:]):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and obj.get("term"):
                recovered.append(obj)
        except Exception:
            pass
    if recovered:
        print(f"[Partial recovery: {len(recovered)} objects]")
    return recovered

def describe_term(term: str):
    """Fallback: get summary+detail for a single term."""
    prompt = (
        f'Define the tech term "{term}". '
        'Reply with ONLY this JSON (no markdown): '
        '{"summary":"one sentence under 20 words","detail":"3 to 6 sentences covering what it is, how it works, why it matters, and common use cases"}'
    )
    try:
        raw = ollama_generate(prompt, max_tokens=300)
        start = raw.find("{"); end = raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(raw[start:end])
            return obj.get("summary", ""), obj.get("detail", "")
    except Exception as e:
        print(f"[describe_term error] {e}")
    return "", ""

# ── Main analysis ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You extract specific tech terms from spoken transcripts and return JSON only.

RULES:
- Only include terms explicitly spoken in the transcript
- Skip generic words (language, framework, database, technology, solution, approach, management, simplification) unless a SPECIFIC named one is mentioned
- DO include: product names, protocols, platforms, DevOps concepts, cloud services, named architectures, specific tools
- Each term needs:
  "summary": 1 sentence max 20 words — what it is at a glance
  "detail": 3 to 6 sentences — cover what it is, how it works, why it matters, and real-world use cases
- Return ONLY a valid JSON array — no explanation, no markdown

OUTPUT FORMAT:
[{"term":"<term>","category":"<tool|framework|language|protocol|concept|cloud|database|security|pattern>","summary":"<summary>","detail":"<detail>"}]

If no specific tech terms: []"""

def analyze(transcript: str) -> list:
    global card_id_counter

    if not transcript or len(transcript) < 20:
        return []

    # Build prompt — use explicit string concat to avoid f-string brace issues
    prompt = SYSTEM_PROMPT + "\n\nTRANSCRIPT:\n" + transcript + "\n\nJSON array:"

    try:
        raw   = ollama_generate(prompt)
        print(f"[Ollama] {raw[:120]}...")
        terms = extract_json_array(raw)
    except Exception as e:
        print(f"[Ollama error] {e}")
        return []

    cards = []
    for t in terms:
        if not isinstance(t, dict):
            continue
        term = t.get("term", "").strip()
        if not term:
            continue
        key = term.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)

        summary = t.get("summary", "").strip()
        detail  = t.get("detail",  "").strip()

        # Fallback if model skipped descriptions
        if not summary or not detail:
            summary, detail = describe_term(term)

        card_id_counter += 1
        cards.append({
            "id":        card_id_counter,
            "term":      term,
            "category":  t.get("category", "concept").strip(),
            "summary":   summary or f"A technical term: {term}",
            "detail":    detail  or "",
            "timestamp": time.strftime("%H:%M:%S"),
        })

    print(f"[Cards: {len(cards)}]")
    return cards

# ── SSE broadcast ─────────────────────────────────────────────
def broadcast(event: str, data: dict):
    msg  = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in sse_clients:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            sse_clients.remove(q)
        except ValueError:
            pass

# ── Audio loop ────────────────────────────────────────────────
def audio_loop():
    idx = get_blackhole_device()
    if idx is None:
        broadcast("error", {"message": "BlackHole not found. Install it and set it as Mac output."})
        state["listening"] = False
        state["status"]    = "error"
        return

    dev      = sd.query_devices(idx)
    channels = min(2, int(dev["max_input_channels"]))
    chunk_n  = int(SAMPLE_RATE * CHUNK_DURATION)
    overlap_n = int(SAMPLE_RATE * OVERLAP)
    buf      = np.zeros(0, dtype=np.float32)
    seen_tx  = set()

    broadcast("status", {"status": "listening", "device": dev["name"]})
    state["status"] = "listening"
    print(f"[Audio] Capturing from: {dev['name']}")

    def cb(indata, frames, t, status):
        nonlocal buf
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buf  = np.concatenate([buf, mono])

    with sd.InputStream(device=idx, samplerate=SAMPLE_RATE,
                        channels=channels, dtype="float32", callback=cb):
        while state["listening"]:
            if len(buf) >= chunk_n:
                chunk = buf[:chunk_n].copy()
                buf   = buf[chunk_n - overlap_n:]

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                print(f"[Audio] RMS={rms:.4f}", end="\r")

                if rms < 0.001:
                    time.sleep(0.1)
                    continue

                print()
                tx = transcribe(chunk)
                print(f"[TX] {tx}")

                if tx and tx not in seen_tx:
                    seen_tx.add(tx)
                    if len(seen_tx) > 100:
                        seen_tx = set(list(seen_tx)[-50:])

                    broadcast("transcript", {"text": tx})

                    for card in analyze(tx):
                        state["cards"].insert(0, card)
                        broadcast("card", card)

            time.sleep(0.2)

    state["status"] = "idle"
    broadcast("status", {"status": "idle"})

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start():
    global audio_thread
    if state["listening"]:
        return jsonify({"ok": False, "message": "Already listening"})
    state["listening"] = True
    state["status"]    = "starting"
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def stop():
    state["listening"] = False
    return jsonify({"ok": True})

@app.route("/api/clear", methods=["POST"])
def clear():
    state["cards"] = []
    seen_terms.clear()
    broadcast("clear", {})
    return jsonify({"ok": True})

@app.route("/api/cards")
def cards():
    return jsonify(state["cards"])

@app.route("/api/status")
def status():
    return jsonify({
        "status":     state["status"],
        "listening":  state["listening"],
        "card_count": len(state["cards"]),
    })

@app.route("/api/devices")
def devices():
    devs      = [{"index": i, "name": d["name"]}
                 for i, d in enumerate(sd.query_devices())
                 if d["max_input_channels"] > 0]
    bh        = get_blackhole_device()
    ok        = False
    models    = []
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
            ok     = True
    except Exception:
        pass
    return jsonify({
        "devices":         devs,
        "blackhole_index": bh,
        "ollama_ok":       ok,
        "ollama_model":    OLLAMA_MODEL,
        "ollama_models":   models,
        "whisper_model":   WHISPER_MODEL_SIZE,
    })

@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=100)
    sse_clients.append(q)
    def generate():
        yield f"event: init\ndata: {json.dumps({'cards': state['cards'], 'status': state['status']})}\n\n"
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                sse_clients.remove(q)
            except ValueError:
                pass
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "your-mac-ip"
    print(f"\n📡 TechRadar")
    print(f"   Whisper : {WHISPER_MODEL_SIZE}")
    print(f"   Ollama  : {OLLAMA_MODEL}")
    print(f"\n   Local  : http://localhost:5001")
    print(f"   Phone  : http://{local_ip}:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
