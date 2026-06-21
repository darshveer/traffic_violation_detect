#!/bin/bash
# Launch the Traffic Violation Detection web app.
#   ./run.sh            # http://localhost:8000
#   PORT=9000 ./run.sh
set -e
cd "$(dirname "$0")"
PYBIN="${PYBIN:-python3}"

# Reuse the project venv if present (has torch/ultralytics/easyocr already),
# else create a local one and install requirements.
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
elif [ -x "../cctv_system/venv/bin/python" ]; then
  PY="../cctv_system/venv/bin/python"
  echo "Using ../cctv_system/venv — installing web deps if missing…"
  "$PY" -m pip install -q fastapi "uvicorn[standard]" python-multipart yt-dlp certifi 2>/dev/null || true
else
  echo "Creating local venv and installing requirements (one-time, downloads ~2GB)…"
  "$PYBIN" -m venv venv
  ./venv/bin/pip install -U pip
  ./venv/bin/pip install -r requirements.txt
  PY="venv/bin/python"
fi

export SSL_CERT_FILE="$("$PY" -c 'import certifi;print(certifi.where())' 2>/dev/null || true)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
echo "Open http://localhost:${PORT:-8000}"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
