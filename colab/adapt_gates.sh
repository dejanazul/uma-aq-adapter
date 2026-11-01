#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
DATA=/content/prep/data; CK=/content/ckpt; DRIVE=/content/drive/MyDrive/aqcat25-lab-data
mkdir -p $CK
export HF_TOKEN=$(cat /content/hf_token)
echo 'STAGE pip'
python -m pip install -q fairchem-core==2.23.0
python -c 'import torch, fairchem.core; print("torch", torch.__version__, "fairchem", fairchem.core.__version__)'
echo 'STAGE checkpoint'
python - <<EOF
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download('facebook/UMA', 'checkpoints/uma-s-1p2.pt', cache_dir='/content/hf_cache'), '$CK/uma-s-1p2.pt')
EOF
sha256sum $CK/uma-s-1p2.pt | tee $CK/uma-s-1p2.sha256
echo 'STAGE gate_s0'
python /content/adapt_surgery.py --in $CK/uma-s-1p2.pt --out $CK/uma-s-1p2-aqcat25-s0.pt --check /content/s0_frames.extxyz
rm $CK/uma-s-1p2-aqcat25-s0.pt
echo 'STAGE surgery'
python /content/adapt_surgery.py --in $CK/uma-s-1p2.pt --out $CK/uma-s-1p2-aqcat25.pt --elem-refs $DATA/aq_elem_refs.json
sha256sum $CK/uma-s-1p2-aqcat25.pt | tee $CK/uma-s-1p2-aqcat25.sha256
echo 'STAGE gate_s1'
python /content/adapt_gate_s1.py $CK/uma-s-1p2.pt $DATA/oc20_val $CK/gate_s1.json 1000
echo 'STAGE drive'
mkdir -p $DRIVE/ckpt
cp $CK/uma-s-1p2-aqcat25.pt $CK/*.sha256 $CK/gate_s1.json $DRIVE/ckpt/
ls -la $DRIVE/ckpt
echo 'STAGE complete'
