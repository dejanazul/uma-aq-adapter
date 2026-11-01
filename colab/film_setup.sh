#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
cd /content
echo 'STAGE uv'
python -m pip install -q uv
uv venv -q --clear -p 3.10 /content/film_venv
PY=/content/film_venv/bin/python
echo 'STAGE torch'
uv pip install -q -p $PY torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
echo 'STAGE pyg'
uv pip install -q -p $PY torch_geometric
uv pip install -q -p $PY torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo 'STAGE deps'
uv pip install -q -p $PY 'numpy<2' e3nn submitit torchtnt hydra-core pymatgen ase orjson wandb tensorboard lmdb \
    huggingface-hub numba datasets pandas tqdm requests
echo 'STAGE fairchem'
rm -rf /content/fairchem
git clone -q https://github.com/facebookresearch/fairchem.git /content/fairchem
git -C /content/fairchem checkout -q tags/fairchem_core-1.10.0
HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 $PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('SandboxAQ/aqcat25-ev2', revision='2a7d5574aa7f0addc0678080eb7621a47a892abd',
                  allow_patterns=['checkpoints_aqcat_ev2/ev2-in+midFiLM*', 'checkpoints_aqcat_ev2/ev2-inFiLM*',
                                  'ev2_film/*', 'patched_code/*'], local_dir='/content/aqcat25-ev2')
EOF
cp /content/aqcat25-ev2/ev2_film/equiformer_v2_film.py /content/fairchem/packages/fairchem-core/src/fairchem/core/models/equiformer_v2/
cp /content/aqcat25-ev2/patched_code/ase_utils.py /content/fairchem/packages/fairchem-core/src/fairchem/core/common/relaxation/ase_utils.py
uv pip install -q -p $PY -e /content/fairchem/packages/fairchem-core --no-deps
grep -q 'def patched_calc' $($PY -c 'import importlib.util as u; print(u.find_spec("fairchem.core").submodule_search_locations[0])')/common/relaxation/ase_utils.py
echo 'STAGE versions'
$PY -c 'import torch, torch_scatter, fairchem.core, ase, numpy; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "fairchem", fairchem.core.__version__, "ase", ase.__version__, "numpy", numpy.__version__)'
uv pip freeze -p $PY > /content/film_pip_freeze.txt
echo "fairchem_commit $(git -C /content/fairchem rev-parse HEAD)"
echo 'STAGE complete'
