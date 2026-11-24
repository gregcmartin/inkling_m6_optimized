#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root="${INKLING_HOME:-$HOME/.inkling-m6-optimized}"
model="Inkling-2-Mini-oQ4e-mtp"
revision="3eea042751cd296b7266628a8342c1749242e1fe"
wheel="omlx-0.8.3-cp311-cp311-macosx_15_0_universal2.whl"
wheel_url="https://github.com/jundot/omlx/releases/download/v0.8.3/$wheel"
wheel_sha="e43afa52bbb19d557ab9690992169431ea9e48d4548be8583b953461d24a615f"
patch="$repo/patches/inkling-fused-omlx-0.8.3.patch"

if [ "$(uname -s):$(uname -m)" != "Darwin:arm64" ]; then
    echo "Run setup in a native arm64 shell on an Apple Silicon Mac." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first (brew install uv)." >&2
    exit 1
fi

umask 077
mkdir -p "$root/profile" "$root/models" "$root/downloads"
wheel_file="$root/downloads/$wheel"
verify_wheel() {
    printf '%s  %s\n' "$wheel_sha" "$1" | shasum -a 256 -c --status
}
if [ ! -f "$wheel_file" ] || ! verify_wheel "$wheel_file"; then
    partial=$(mktemp "$root/downloads/.wheel.XXXXXXXX")
    trap 'rm -f "$partial"' EXIT
    curl --fail --location --retry 3 --output "$partial" "$wheel_url"
    verify_wheel "$partial"
    mv "$partial" "$wheel_file"
    trap - EXIT
fi
if [ ! -x "$root/venv/bin/python" ]; then
    uv venv --python 3.11 "$root/venv"
fi
python="$root/venv/bin/python"
uv pip install --python "$python" "$wheel_file"
site=$("$python" -B -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

overlay="$root/fused-runtime"
if git -C "$overlay" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Fused Inkling kernels are already installed."
else
    staged=$(mktemp -d "$root/.fused.XXXXXXXX")
    trap 'rm -rf "$staged"' EXIT
    cp -R "$site/omlx" "$staged/omlx"
    git -C "$staged" apply --check "$patch"
    git -C "$staged" apply "$patch"
    if [ -e "$overlay" ]; then
        echo "Move the existing $overlay aside, then re-run setup." >&2
        exit 1
    fi
    mv "$staged" "$overlay"
    trap - EXIT
fi
for file in settings.json model_settings.json; do
    if [ ! -e "$root/profile/$file" ]; then
        cp "$repo/config/$file" "$root/profile/$file"
    fi
done

HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "$root/venv/bin/hf" download "mlx-community/$model" \
    --revision "$revision" --local-dir "$root/models/$model"
printf '\nStart oMLX with %s/serve.sh\n' "$repo"
