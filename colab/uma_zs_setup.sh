#!/usr/bin/env bash
set -euo pipefail
echo 'STAGE pip'
python -m pip install -q fairchem-core==2.23.0
echo 'STAGE versions'
python -c 'import torch, fairchem.core, ase; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "fairchem", fairchem.core.__version__, "ase", ase.__version__)'
python -m pip freeze > /content/uma_pip_freeze.txt
echo 'STAGE complete'
