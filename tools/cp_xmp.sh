#!/bin/bash

# Copy *.xmp to ./data/xmp0     # for analysis
rsync -avu --prune-empty-dirs \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    kepler:/mnt/d/Lightroom/MediaFiles/Photos/Photos-{18..25} \
    "$HOME/projects/bird_cluster/data/xmp/" --dry-run

rsync -avc --prune-empty-dirs \
    kepler:/mnt/d/_Staging/jpg/ \
    "$HOME/projects/bird_cluster/data/jpg/" --dry-run

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