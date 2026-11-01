#!/usr/bin/env bash
NAME=$1; CONFIG=$2
RUN=/content/runs/$NAME; LOG=/content/$NAME.log; D=/content/drive/MyDrive/aqcat25-lab-data/runs

sync_drive() {
    mkdir -p $D/$NAME $D/${NAME}_resume
    rsync -a --exclude='*.distcp' --exclude='.metadata' $RUN/ $D/$NAME/
    cp $LOG $D/$NAME.log
    latest=$(ls -t $RUN/*/checkpoints/*/inference_ckpt.pt 2>/dev/null | head -1)
    [ -n "$latest" ] || return 0
    rel=$(dirname ${latest#$RUN/})
    mkdir -p $D/${NAME}_resume/$rel
    rsync -a --exclude=inference_ckpt.pt $RUN/$rel/ $D/${NAME}_resume/$rel/
    find $D/${NAME}_resume -mindepth 3 -maxdepth 3 -type d ! -path "*/$rel" -exec rm -rf {} +
}

fairchem -c $CONFIG >> $LOG 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
    sleep 600
    sync_drive >> $LOG 2>&1
done
wait $pid; code=$?
echo "exit $code" >> $LOG
sync_drive >> $LOG 2>&1
echo all_done >> $LOG
