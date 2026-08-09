#!/bin/bash

# Copy *.xmp to ./data/xmp0     # for analysis
rsync -avu \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    /syno/Lightroom/MediaFiles/Photos/Photos-{16..25} \
    "$HOME/projects/bird_cluster/data/xmp/" --dry-run

# Apply xmp to Lightroom
rsync -avc \
    --include='*/' \
    --include='*.xmp' \
    --exclude='*' \
    --existing \
    $HOME/projects/bird_cluster/data/label/raw/Photos-{16..25} \
    /syno/Lightroom/Media\ Files/Photos/ --dry-run


rsync -av --prune-empty-dirs \
    --delete \
    kepler:/mnt/d/_Staging/jpg/ \
    "$HOME/projects/bird_cluster/data/jpg/_flat/" --dry-run

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