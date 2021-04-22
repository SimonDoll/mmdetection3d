#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=${3:-1}
PORT=${PORT:-29555}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/evaluate.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
