#!/usr/bin/env bash
set -euo pipefail
NAME=$1; RUN=/content/runs/$NAME; D=/content/drive/MyDrive/aqcat25-lab-data/runs
mkdir -p $RUN
rsync -a --exclude=inference_ckpt.pt $D/$NAME/ $RUN/
rsync -a $D/${NAME}_resume/ $RUN/
[ -f $D/$NAME.log ] && cp $D/$NAME.log /content/$NAME.log
cfg=$(ls $D/${NAME}_resume/*/checkpoints/*/resume.yaml | head -1)
cfg=$RUN/${cfg#$D/${NAME}_resume/}
echo "resume from $cfg" | tee -a /content/$NAME.log
python - $cfg $RUN/run.yaml <<'EOF2'
import sys
from omegaconf import OmegaConf
r, o = OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2])
m = OmegaConf.to_container(o.runner.train_eval_unit.model, resolve=True)
if m['_target_'] != 'fairchem.core.units.mlip_unit.mlip_unit.initialize_finetuning_model':
    r.runner.train_eval_unit.model = m
    OmegaConf.save(r, sys.argv[1])
    print('resume model ->', m['_target_'])
EOF2
setsid nohup bash /content/adapt_train_job.sh $NAME $cfg > /dev/null 2>&1 &
