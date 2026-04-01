#!/bin/bash
# TechRadar (free/local) — setup & run
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}  📡 TECHRADAR  —  free / local mode${NC}"
echo -e "${CYAN}  Whisper runs on your Mac · Ollama runs on your Mac · \$0/month${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── Python venv ─────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo -e "${GREEN}→ Creating Python virtual environment...${NC}"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo -e "${GREEN}→ Installing Python dependencies...${NC}"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Ollama check ─────────────────────────────────────────────
echo ""
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"

if ! curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
  echo -e "${YELLOW}⚠  Ollama doesn't seem to be running.${NC}"
  echo    "   Start it with:  ollama serve"
  echo    "   Then re-run this script."
  exit 1
else
  echo -e "${GREEN}✓ Ollama is running${NC}"
fi

# Check if the model is pulled
if ! ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
  echo -e "${YELLOW}→ Pulling $OLLAMA_MODEL (one-time download)...${NC}"
  ollama pull "$OLLAMA_MODEL"
else
  echo -e "${GREEN}✓ Ollama model '$OLLAMA_MODEL' ready${NC}"
fi

# ── BlackHole check ──────────────────────────────────────────
python3 - <<'EOF'
import sounddevice as sd
devices = [d['name'].lower() for d in sd.query_devices() if d['max_input_channels'] > 0]
found = any('blackhole' in d or 'loopback' in d or 'soundflower' in d for d in devices)
if found:
    print("\033[0;32m✓ Audio loopback device found\033[0m")
else:
    print("\033[1;33m⚠  BlackHole not detected.\033[0m")
    print("   Install: https://existential.audio/blackhole/")
    print("   Then set BlackHole as Mac output in System Settings → Sound")
EOF

# ── Get local IP ─────────────────────────────────────────────
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "your-mac-ip")

echo ""
echo -e "${GREEN}${BOLD}✓ All set — launching TechRadar${NC}"
echo ""
echo -e "  ${BOLD}Browser (Mac):${NC}  http://localhost:5001"
echo -e "  ${BOLD}Phone:${NC}          http://${LOCAL_IP}:5001"
echo ""
echo -e "  ${YELLOW}Tip: Whisper model downloads ~150MB on first run — normal!${NC}"
echo ""
echo -e "${CYAN}Starting...${NC}"
echo "────────────────────────────────────"

export OLLAMA_MODEL
cd "$SCRIPT_DIR"
python3 app.py
