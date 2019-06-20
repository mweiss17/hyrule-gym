#!/bin/sh

# $1 = video path
# $2 = destination folder

mkdir -p $2
ffmpeg -i $1 -r 30 -map 0:0 "$2/pano_%06d.png"

# Old version for jpgs

#mkdir -p "$2/tmp"
#ffmpeg -i $1 -r 30 -map 0:0 "$2/tmp/pano_%06d.png"
#mogrify -format jpg -quality 90 -path $2  $2/tmp/*.png
#rm -rf "$2/tmp"






