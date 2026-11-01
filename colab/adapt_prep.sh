#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
REV=2ec3a36c138ef0150ac9b1d783a7acf0aadfeda9
HF=https://huggingface.co/datasets/SandboxAQ/aqcat25-dataset/resolve/$REV
WORK=/content/prep; DRIVE=/content/drive/MyDrive/aqcat25-lab-data
mkdir -p $WORK/pq $WORK/data
export HF_TOKEN=$(cat /content/hf_token)
echo 'STAGE pip'
python -m pip install -q ase ase-db-backends pandas pyarrow huggingface_hub requests
python -c 'import ase, ase_db_backends, pandas; print("ase", ase.__version__, "pandas", pandas.__version__)'
echo 'STAGE parquet'
python - <<EOF
from huggingface_hub import hf_hub_download
for f in ('train_id', 'id_slabs'):
    hf_hub_download('SandboxAQ/aqcat25-dataset', f'parquet/{f}.parquet', repo_type='dataset', revision='$REV', local_dir='$WORK/pq_dl')
EOF
mv $WORK/pq_dl/parquet/*.parquet $WORK/pq/
md5sum $WORK/pq/*.parquet
echo 'STAGE select'
python /content/adapt_select.py $WORK/pq/train_id.parquet $WORK/pq/id_slabs.parquet $WORK
echo 'STAGE aq'
python /content/adapt_data.py aq $WORK/data $WORK/selection.csv $HF/train_id.tar.gz $HF/id_slabs.tar.gz
echo 'STAGE oc20'
python /content/adapt_data.py oc20 $WORK/data https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar 5000 953474cb93f0b08cdc523399f03f7c36
echo 'STAGE drive'
du -sh $WORK/data/*
for i in $(seq 1 240); do [ -d /content/drive/MyDrive ] && break; echo "waiting for drive mount ($i)"; sleep 30; done
mkdir -p $DRIVE
cp $WORK/selection.csv $WORK/selection_summary.json $DRIVE/
cp -r $WORK/data $DRIVE/
du -sh $DRIVE/data
echo 'STAGE complete'
