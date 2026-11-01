#!/usr/bin/env bash
set -euo pipefail
SRC=/content/drive/MyDrive/aqcat25-lab-data
echo 'STAGE pip'
python -m pip install -q fairchem-core==2.23.0 ase-db-backends
python -c 'import torch, fairchem.core; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "fairchem", fairchem.core.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m pip freeze > /content/adapt_pip_freeze.txt
echo 'STAGE copy'
mkdir -p /content/data /content/ckpt
rsync -a $SRC/data/ /content/data/
cp $SRC/ckpt/uma-s-1p2-aqcat25*.pt $SRC/ckpt/uma-s-1p2-aqcat25*.sha256 /content/ckpt/
(cd /content/ckpt && cat uma-s-1p2-aqcat25*.sha256 | sha256sum -c)
du -sh /content/data/*
echo 'STAGE complete'
