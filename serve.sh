#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root="${INKLING_HOME:-$HOME/.inkling-m6-optimized}"
python="$root/venv/bin/python"
model="$root/models/Inkling-2-Mini-oQ4e-mtp"
if [ ! -x "$python" ] || [ ! -f "$model/model.safetensors.index.json" ] ||
   [ ! -f "$model/mtp.safetensors" ] || [ ! -f "$root/profile/model_settings.json" ]; then
    echo "Run ./setup.sh to install the runtime, settings, and checkpoint." >&2
    exit 1
fi
case "${INKLING_ENGINE:-fused}" in
    fused)
        overlay="$root/fused-runtime"
        if ! git -C "$overlay" apply --reverse --check "$repo/patches/inkling-fused-omlx-0.8.3.patch" 2>/dev/null; then
            echo "The fused kernel backport is missing. Run ./setup.sh." >&2
            exit 1
        fi
        export PYTHONPATH="$overlay"
        ;;
    stock)
        unset PYTHONPATH
        ;;
    *)
        echo "INKLING_ENGINE must be fused or stock." >&2
        exit 1
        ;;
esac

export OMLX_HEADLESS=1 HF_HUB_OFFLINE=1
exec "$root/venv/bin/omlx" serve --headless --base-path "$root/profile" \
    --model-dir "$root/models" --host 127.0.0.1 --port "${INKLING_PORT:-8000}"
