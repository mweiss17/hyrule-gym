#!/bin/sh
# Extract pngs from mp4. The camera captures at 30 fps.
# $1 = name of the data folder
# $2 = video filename
# $3 = fps

mkdir -p "data/$1/raw/pngs"

ffmpeg -i "data/$1/raw/video/$2" -r $3 -map 0:0 "data/$1/raw/pngs/track_1_output_%06d.png"
ffmpeg -i "data/$1/raw/video/$2" -r $3 -map 0:1 "data/$1/raw/pngs/track_2_output_%06d.png"
