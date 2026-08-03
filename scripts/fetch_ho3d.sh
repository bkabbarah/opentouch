#!/bin/bash
# Fetch HO-3D v3, keep ONLY the annotations, and run the rotation diagnostic.
#
# TWO WAYS TO RUN THIS
#
#   bash scripts/fetch_ho3d.sh --from-curl
#       Reads /scratch/bashar/datasets/ho3d_curl.txt, which holds a
#       "Copy as cURL (bash)" captured from Chrome DevTools. USE THIS.
#
#   bash scripts/fetch_ho3d.sh "<url>"
#       Only works if the URL carries its own auth token. OneDrive anonymous
#       shares do not: copying the link address drops the session cookies that
#       redeem the share, and the bare download.aspx form answers 403.
#
# Either path: downloads the zip, extracts only meta/*.pkl, VERIFIES the
# extraction before deleting the 31.9 GB zip, converts to (N,T,21,3), and runs
# scripts/rotation_share.py.

set -uo pipefail
D=/scratch/bashar/datasets/ho3d
REPO=~/scratch/bashar/opentouch-gru
PY=~/miniconda3/envs/opentouch/bin/python
CURLFILE=/scratch/bashar/datasets/ho3d_curl.txt
mkdir -p "$D"
cd "$D" || exit 1

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "ERROR: pass --from-curl (recommended) or a URL in quotes."
    echo "See the comment block at the top of this script."
    exit 1
fi

print_curl_recipe() {
    echo "How to capture a working request:"
    echo "  1. Chrome: open the HO3D_v3 OneDrive folder"
    echo "  2. Press F12, select the Network tab"
    echo "  3. Click HO3D_v3.zip, then Download. A request appears in Network."
    echo "  4. Right-click that request -> Copy -> Copy as cURL (bash)"
    echo "  5. Cancel the Chrome download; you do not want 31.9 GB on your laptop"
    echo "  6. On the cluster:"
    echo "         cat > $CURLFILE"
    echo "     paste, then press Ctrl-D"
    echo "  7. bash scripts/fetch_ho3d.sh --from-curl"
}

echo "== [1/4] downloading HO3D_v3.zip (31.9 GB) -- resumable, safe to re-run =="
if [ "$MODE" = "--from-curl" ]; then
    if [ ! -s "$CURLFILE" ]; then
        echo "ERROR: $CURLFILE is missing or empty."
        print_curl_recipe
        exit 1
    fi
    # Drop any output/compression flags Chrome included, then add ours. The
    # cookies and headers that make the anonymous share work are preserved.
    CMD=$(tr -d "\n" < "$CURLFILE" \
        | sed -E "s/[[:space:]]--output[[:space:]]+[^[:space:]]+//g; \
                  s/[[:space:]]-o[[:space:]]+[^[:space:]]+//g; \
                  s/[[:space:]]--compressed//g")
    eval "$CMD --fail --location --retry 5 --retry-delay 15 --continue-at - --output HO3D_v3.zip"
    RC=$?
else
    curl -fL --retry 5 --retry-delay 15 -C - -o HO3D_v3.zip "$MODE"
    RC=$?
fi

if [ $RC -ne 0 ]; then
    echo
    echo "DOWNLOAD FAILED (curl exit $RC). Nothing was deleted."
    if [ "$MODE" != "--from-curl" ]; then
        echo "A 403 or 401 here means the URL had no auth token."
        print_curl_recipe
    fi
    exit 1
fi

SZ=$(stat -c %s HO3D_v3.zip 2>/dev/null || echo 0)
if [ "$SZ" -lt 1000000000 ]; then
    echo "DOWNLOAD TOO SMALL ($SZ bytes) -- that is an error page, not the zip."
    echo "Keeping it. First bytes:"
    head -c 300 HO3D_v3.zip
    echo
    exit 1
fi

echo "== [2/4] extracting annotations only (meta/*.pkl) =="
unzip -q -o HO3D_v3.zip "*/meta/*" -d extracted
NPKL=$(find extracted -name "*.pkl" | wc -l)
echo "   extracted $NPKL annotation files"

# Dataset root is the directory CONTAINING train/, whether or not the zip
# nests everything under an HO3D_v3/ top level. Both layouts verified.
TRAINDIR=$(find "$D/extracted" -type d -name train | head -1)
if [ -z "$TRAINDIR" ] || [ "$NPKL" -lt 1000 ]; then
    echo "ERROR: extraction did not produce the expected layout."
    echo "  .pkl files: $NPKL ; train/ dir: ${TRAINDIR:-none}"
    echo "  KEEPING HO3D_v3.zip so you do not re-download 31.9 GB."
    echo "  Inspect with: unzip -l $D/HO3D_v3.zip | head -30"
    exit 1
fi
ROOT=$(dirname "$TRAINDIR")
echo "   dataset root: $ROOT"

echo "== extraction verified; deleting the zip to reclaim 31.9 GB =="
rm -f HO3D_v3.zip

echo "== [3/4] converting to (N,T,21,3) =="
cd "$REPO" || exit 1
PYTHONPATH=src $PY scripts/load_public_hands.py ho3d --root "$ROOT" --out-prefix "$D/ho3d" || exit 1

echo "== [4/4] rotation-share diagnostic =="
PYTHONPATH=src $PY scripts/rotation_share.py --npy "$D/ho3d_poses.npy" \
    --groups-npy "$D/ho3d_groups.npy" --fps 30 --name HO3D \
    --out results/results_rotation_share_ho3d.json || exit 1

echo
echo "DONE. Compare the k=8 median against OpenTouch 0.961."
