#!/bin/bash

# Copy *.xmp to ./data/xmp0     # for analysis
rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    --delete \
    Photos/Photos-{18..25} \
    "spark:$HOME/projects/bird_cluster/data/xmp/"

# Copy the photo library
rsync -av \
    --include='Photos-18/***' \
    --include='Photos-19/***' \
    --include='Photos-20/***' \
    --include='Photos-21/***' \
    --include='Photos-22/***' \
    --include='Photos-23/***' \
    --include='Photos-24/***' \
    --include='Photos-25/***' \
    --exclude='*' \
    Photos/ \
    "kepler:/mnt/d/Lightroom/MediaFiles/Photos"

#Backup the photo library before mvong photos around
rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    Photos/Photos-{18..25} \
    "n4:/syno/Lightroom/MediaFiles/Photos/"