#!/usr/bin/env bash
set -uo pipefail
bash /content/adapt_gpu_setup.sh || exit 1
export ADAPT_CKPT=/content/ckpt/uma-s-1p2-aqcat25-r2.pt ADAPT_REFS=/content/data/aq_elem_refs_r2.json
COMMON="runner.max_steps=4000 evaluate_every_n_steps=1000 checkpoint_every_n_steps=4000"
wait_arm() {
    for i in $(seq 1 720); do grep -q all_done /content/$1.log 2>/dev/null && return 0; sleep 10; done
    echo "arm $1 timed out"; return 1
}
cd /content
echo "STAGE u1_lr"
ADAPT_REPLAY=1 python adapt_launch.py B u1_lr 110000 8 $COMMON lr=0.00004 && wait_arm u1_lr
echo "STAGE u1_replay"
ADAPT_REPLAY=3 python adapt_launch.py B u1_replay 110000 8 $COMMON runner.train_eval_unit.cosine_lr_scheduler_fn.warmup_epochs=0.015 && wait_arm u1_replay
echo "STAGE complete"
