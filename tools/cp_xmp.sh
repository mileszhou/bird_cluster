#!/bin/bash

# Copy *.xmp to ./data/xmp0     # for analysis
rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    Photos/Photos-{18..25} \
    "spark:$HOME/projects/bird_cluster/data/xmp0/"