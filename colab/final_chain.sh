#!/usr/bin/env bash
set -uo pipefail
SRC=/content/drive/MyDrive/aqcat25-lab-data
if [ -z "${SKIP_SETUP:-}" ]; then
    echo 'STAGE pip'
    python -m pip install -q fairchem-core==2.23.0 ase-db-backends || exit 1
    python -m pip freeze > /content/final_pip_freeze.txt
    echo 'STAGE copy'
    mkdir -p /content/ckpt /content/data
    cp $SRC/ckpt/uma-s-1p2-aqcat25-r2.pt $SRC/ckpt/uma-s-1p2-aqcat25-r2.sha256 /content/ckpt/ || exit 1
    (cd /content/ckpt && sha256sum -c uma-s-1p2-aqcat25-r2.sha256) || exit 1
    rsync -a $SRC/data/aq_val /content/data/ || exit 1
fi
cd /content
export PYTHONPATH=/content HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1
for step in ${STEPS:-gate aq aqz oc20 forget}; do
    echo "STAGE $step"
    python final_eval.py $step || { echo "STEP FAILED $step"; exit 1; }
    mkdir -p $SRC/final && cp /content/final/* $SRC/final/
done
echo 'STAGE final_complete'
