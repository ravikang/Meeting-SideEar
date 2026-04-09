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
WHISPER_MODEL_SIZE  = os.environ.get("WHISPER_MODEL", "medium")
OLLAMA_URL          = os.environ.get("OLLAMA_URL",    "http://localhost:11434")
OLLAMA_MODEL        = os.environ.get("OLLAMA_MODEL",  "llama3.2")
SAMPLE_RATE         = 16000
PREVIEW_CHUNK_SECS  = 2   # Pipeline A: live transcript cadence
ANALYSIS_CHUNK_SECS = 8   # Pipeline B: Ollama term detection cadence
OVERLAP_SECS        = 1

# ── Load Whisper ──────────────────────────────────────────────
print(f"Loading Whisper '{WHISPER_MODEL_SIZE}'...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("✓ Whisper ready")

# ── Resolve Ollama model ──────────────────────────────────────
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
    "listening": False,
    "cards":     [],
    "status":    "idle",
}
seen_terms      = set()
sse_clients     = []
audio_thread    = None
card_id_counter = 0

# ── Focused taxonomy ──────────────────────────────────────────
FOCUS_DOMAINS = """
DOMAINS TO DETECT (only catch terms from these areas):

AI & Machine Learning:
  AI, ML, deep learning, neural network, NLP, computer vision, foundation model,
  large language model, LLM, generative AI, gen AI, RAG, retrieval augmented generation,
  prompt engineering, fine-tuning, embeddings, vector database, AI agent, agentic AI,
  reinforcement learning, transformer, diffusion model, multimodal AI, watsonx,
  IBM watsonx, Watson, IBM Watson

AI & Data Governance:
  AI governance, model risk management, explainability, bias detection, model drift,
  hallucination, responsible AI, trustworthy AI, AI ethics, fairness, transparency,
  auditability, SR 11-7, model validation, OpenScale, IBM OpenPages, OpenPages,
  data governance, data lineage, data catalog, metadata management, master data management,
  MDM, data quality, data stewardship, data mesh, data fabric

Automation & Orchestration:
  RPA, robotic process automation, intelligent automation, hyperautomation,
  workflow orchestration, business process automation, BPA, AIOps, MLOps, LLMOps,
  event-driven architecture, EDA, iPaaS, IBM Cloud Pak, Cloud Pak for Integration,
  Cloud Pak for Data, API management, API gateway

Data & Analytics:
  data lakehouse, data lake, data warehouse, data platform, DataOps,
  ETL, ELT, CDC, change data capture, data pipeline, data streaming,
  Apache Kafka, Spark, IBM Db2, Db2, IBM InfoSphere, InfoSphere,
  real-time analytics, business intelligence, BI, predictive analytics,
  IBM Cognos, Cognos, data virtualization, federated data

Cloud & Infrastructure:
  hybrid cloud, multi-cloud, IBM Cloud, Red Hat, OpenShift, Red Hat OpenShift,
  Kubernetes, containerization, Docker, bare metal, edge computing,
  IBM Z, mainframe, IBM Power, cloud-native, serverless, microservices,
  service mesh, Istio, infrastructure as code, IaC, Terraform, Ansible

Networking & Security:
  zero trust, SASE, SD-WAN, IBM QRadar, QRadar, SOAR, SIEM,
  threat intelligence, identity and access management, IAM,
  privileged access management, PAM, IBM MQ, MQ messaging,
  encryption, tokenization, data masking, DLP, data loss prevention,
  cyber resilience, incident response

Banking & Financial Services:
  core banking, payment rails, SWIFT, ISO 20022, open banking, PSD2,
  anti-money laundering, AML, know your customer, KYC,
  credit risk modeling, fraud detection, real-time payments,
  Basel III, Basel IV, DORA, regulatory reporting, stress testing,
  financial crime, sanctions screening, collateral management

Integration & Middleware:
  IBM MQ, IBM App Connect, App Connect, API Connect, IBM API Connect,
  event streaming, message broker, ESB, enterprise service bus,
  service-oriented architecture, SOA, IBM Integration Bus
"""

BLOCKLIST = {
    "strategy", "approach", "solution", "management", "simplification",
    "transformation", "initiative", "journey", "framework", "platform",
    "technology", "technologies", "tool", "tools", "system", "systems",
    "process", "processes", "capability", "capabilities", "environment",
    "environments", "workload", "workloads", "infrastructure", "layer",
    "layers", "component", "components", "service", "services",
    "devops", "agile", "digital", "innovation", "ecosystem", "landscape",
    "architecture", "integration", "automation", "analytics", "data",
    "cloud", "security", "network", "application", "applications",
}

# ── Audio helpers ─────────────────────────────────────────────
def get_blackhole_device():
    for i, d in enumerate(sd.query_devices()):
        if any(k in d["name"].lower() for k in ("blackhole", "loopback", "soundflower")):
            if d["max_input_channels"] > 0:
                return i
    return None

def transcribe(audio: np.ndarray, beam_size: int = 3) -> str:
    if not len(audio):
        return ""
    audio = audio.astype(np.float32)
    peak  = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    try:
        segs, _ = whisper_model.transcribe(
            audio, language="en", beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        return " ".join(s.text for s in segs).strip()
    except Exception as e:
        print(f"[Whisper error] {e}")
        return ""

# ── Ollama helpers ────────────────────────────────────────────
def ollama_generate(prompt: str, max_tokens: int = 2000) -> str:
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
    raw = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("[")
    if start == -1:
        return []
    end = raw.rfind("]")
    if end > start:
        try:
            result = json.loads(raw[start:end+1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
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
    prompt = (
        'You are an expert in IBM technology, AI, data, banking, and cloud infrastructure. '
        f'Define the term "{term}" as used in enterprise AI, data, or banking contexts. '
        'Reply with ONLY this JSON (no markdown): '
        '{"summary":"one sentence under 20 words","detail":"3 to 6 sentences: what it is, '
        'how it works, why it matters in banking or enterprise AI, and IBM-specific context if relevant"}'
    )
    try:
        raw = ollama_generate(prompt, max_tokens=400)
        start = raw.find("{"); end = raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(raw[start:end])
            return obj.get("summary", ""), obj.get("detail", "")
    except Exception as e:
        print(f"[describe_term error] {e}")
    return "", ""

def is_likely_persons_name(term: str) -> bool:
    words = term.strip().split()
    if len(words) == 1:
        w = words[0]
        if len(w) > 1 and w[0].isupper() and w[1:].islower() and len(w) < 12:
            return True
    return False

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a specialist term extractor for enterprise meetings about AI, data, banking, "
    "IBM technology, cloud, and IT infrastructure.\n\n"
    "YOUR JOB: Read the transcript and extract ONLY terms that appear in the DOMAINS list below.\n\n"
    + FOCUS_DOMAINS +
    "\nSTRICT RULES:\n"
    "1. ONLY extract terms explicitly spoken in the transcript — never invent or infer\n"
    "2. NEVER extract people's names, company names, or place names\n"
    "3. NEVER extract generic words like: strategy, approach, solution, journey, transformation, "
    "workload, environment, layer, component, landscape, ecosystem, platform, tool, process\n"
    "4. A term must match one of the domains above to qualify\n"
    "5. Prefer specific named products/protocols over generic concepts\n"
    "6. Each term needs:\n"
    '   "summary": 1 sentence max 20 words\n'
    '   "detail": 3-6 sentences covering what it is, how it works, '
    "why it matters in banking/enterprise AI, IBM context where relevant\n"
    "7. Return ONLY a valid JSON array — no prose, no markdown\n\n"
    "OUTPUT FORMAT:\n"
    '[{"term":"<term>","category":"<ai|llm|governance|automation|data|cloud|security|banking|integration>","summary":"<summary>","detail":"<detail>"}]\n\n'
    "If no qualifying terms found: []"
)

def analyze(transcript: str) -> list:
    global card_id_counter
    if not transcript or len(transcript) < 20:
        return []

    prompt = SYSTEM_PROMPT + "\n\nTRANSCRIPT:\n" + transcript + "\n\nJSON array:"

    try:
        raw   = ollama_generate(prompt)
        print(f"[Ollama] {raw[:150]}...")
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
        if term.lower() in BLOCKLIST:
            print(f"[Filtered - blocklist] {term}")
            continue
        if is_likely_persons_name(term):
            print(f"[Filtered - name] {term}")
            continue
        key = term.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)

        summary = t.get("summary", "").strip()
        detail  = t.get("detail",  "").strip()
        if not summary or not detail:
            summary, detail = describe_term(term)

        card_id_counter += 1
        cards.append({
            "id":        card_id_counter,
            "term":      term,
            "category":  t.get("category", "concept").strip(),
            "summary":   summary or f"{term} — a term from enterprise AI or banking technology.",
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

# ── Pipeline B: analysis worker thread ───────────────────────
def analysis_worker(analysis_queue):
    """
    Consumes 8-second audio chunks from the queue, transcribes them
    with high-accuracy beam search, then sends to Ollama for term detection.
    Runs in its own thread so it never blocks Pipeline A.
    """
    while state["listening"] or not analysis_queue.empty():
        try:
            chunk = analysis_queue.get(timeout=1)
        except queue.Empty:
            continue
        print("\n[Pipeline B] Transcribing chunk for analysis...")
        tx = transcribe(chunk, beam_size=5)
        print(f"[Pipeline B TX] {tx}")
        if tx:
            for card in analyze(tx):
                state["cards"].insert(0, card)
                broadcast("card", card)

# ── Pipeline A + B coordinator ────────────────────────────────
def audio_loop():
    """
    Main audio capture loop. Maintains two independent pipelines:

    Pipeline A (fast) — reads every 2s, beam_size=1 → live transcript
    Pipeline B (slow) — accumulates 8s, beam_size=5 → Ollama term cards

    Both pipelines share the same input stream.
    """
    idx = get_blackhole_device()
    if idx is None:
        broadcast("error", {"message": "BlackHole not found. Install it and set it as Mac output."})
        state["listening"] = False
        state["status"]    = "error"
        return

    dev       = sd.query_devices(idx)
    channels  = min(2, int(dev["max_input_channels"]))

    preview_n  = int(SAMPLE_RATE * PREVIEW_CHUNK_SECS)
    analysis_n = int(SAMPLE_RATE * ANALYSIS_CHUNK_SECS)
    overlap_n  = int(SAMPLE_RATE * OVERLAP_SECS)

    # Mutable buffers stored in a dict so the callback can update them
    buffers = {
        "audio":    np.zeros(0, dtype=np.float32),
        "analysis": np.zeros(0, dtype=np.float32),
    }
    seen_tx        = set()
    analysis_queue = queue.Queue(maxsize=4)

    broadcast("status", {"status": "listening", "device": dev["name"]})
    state["status"] = "listening"
    print(f"[Audio] Capturing from: {dev['name']}")
    print(f"[Pipelines] A={PREVIEW_CHUNK_SECS}s transcript  B={ANALYSIS_CHUNK_SECS}s analysis")

    # Start Pipeline B worker thread
    worker = threading.Thread(
        target=analysis_worker, args=(analysis_queue,), daemon=True
    )
    worker.start()

    def cb(indata, frames, t, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buffers["audio"] = np.concatenate([buffers["audio"], mono])

    with sd.InputStream(device=idx, samplerate=SAMPLE_RATE,
                        channels=channels, dtype="float32", callback=cb):
        while state["listening"]:

            if len(buffers["audio"]) >= preview_n:
                # Grab a preview chunk
                chunk_a          = buffers["audio"][:preview_n].copy()
                buffers["audio"] = buffers["audio"][preview_n:]

                rms = float(np.sqrt(np.mean(chunk_a ** 2)))
                print(f"[Audio] RMS={rms:.4f}", end="\r")

                # Pipeline A — fast live transcript (beam=1 for speed)
                if rms >= 0.001:
                    tx = transcribe(chunk_a, beam_size=1)
                    if tx and tx not in seen_tx:
                        seen_tx.add(tx)
                        if len(seen_tx) > 200:
                            seen_tx = set(list(seen_tx)[-100:])
                        print(f"\n[A] {tx}")
                        broadcast("transcript", {"text": tx})

                # Accumulate for Pipeline B
                buffers["analysis"] = np.concatenate([buffers["analysis"], chunk_a])

                # Pipeline B — hand off when we have a full analysis chunk
                if len(buffers["analysis"]) >= analysis_n:
                    chunk_b             = buffers["analysis"][:analysis_n].copy()
                    buffers["analysis"] = buffers["analysis"][analysis_n - overlap_n:]
                    if not analysis_queue.full():
                        analysis_queue.put_nowait(chunk_b)
                    else:
                        print("\n[B] Queue full — dropping chunk")

            time.sleep(0.05)

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
    devs   = [{"index": i, "name": d["name"]}
               for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0]
    bh     = get_blackhole_device()
    ok     = False
    models = []
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
    print(f"\n👂 Meeting Side-Ear")
    print(f"   Whisper  : {WHISPER_MODEL_SIZE}")
    print(f"   Ollama   : {OLLAMA_MODEL}")
    print(f"   Pipeline A (transcript) : every {PREVIEW_CHUNK_SECS}s")
    print(f"   Pipeline B (term cards) : every {ANALYSIS_CHUNK_SECS}s")
    print(f"\n   Local  : http://localhost:5001")
    print(f"   Phone  : http://{local_ip}:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
