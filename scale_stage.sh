#!/bin/bash
set -e
STAGE=$1
REPLICAS=$2

if [ "$STAGE" == "detection" ]; then
    kubectl scale deployment/cell-detection --replicas=$REPLICAS
elif [ "$STAGE" == "tracking" ]; then
    kubectl scale deployment/cell-tracking --replicas=$REPLICAS
else
    echo "Usage: ./scale_stage.sh [detection|tracking] [REPLICA_COUNT]"
    exit 1
fi
echo "Scaled $STAGE to $REPLICAS replicas."
