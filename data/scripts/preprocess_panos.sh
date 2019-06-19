#!/bin/sh

# $1 = video path
# $2 = destination folder

# mkdir -p "$2/pngs"

# ffmpeg -i $1 -r 30 -map 0:0 "$2/pngs/track_1_output_%06d.png"
# ffmpeg -i $1 -r 30 -map 0:1 "$2/pngs/track_2_output_%06d.png"
# python scripts/crop_parallel.py $2/pngs crops

# rm -rf $2/pngs

# python scripts/stitch_panos.py $2/crops

mkdir -p "$2/panos_jpg"
mogrify -format jpg -quality 90 -path $2/panos_jpg  $2/panoramas/*.png

# rm -rf $2/panoramas





