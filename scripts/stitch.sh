#!/bin/sh
# $1 = input dir
# $2 = output dir
# $3 = frame num
echo "$1/camera_1_frame_$3.png"

pto_gen -o "pano_frame_$3.pto" -p 2 -f 120 "$1/camera_1_frame_$3.png" "$1/camera_3_frame_$3.png" "$1/camera_5_frame_$3.png" "$1/camera_7_frame_$3.png"
cpfind -o "pano_frame_$3.pto" --multirow --celeste "pano_frame_$3.pto"
cpclean -o "pano_frame_$3.pto" "pano_frame_$3.pto"
linefind -o "pano_frame_$3.pto" "pano_frame_$3.pto"
autooptimiser -a -m -l -s -o "pano_frame_$3.pto" "pano_frame_$3.pto"
pano_modify --canvas=AUTO --crop=AUTO -o "pano_frame_$3.pto" "pano_frame_$3.pto"
hugin_executor --stitching --prefix="$2/pano_frame_$3" "pano_frame_$3.pto"
convert "$2/pano_frame_$3.tif" "$2/pano_frame_$3.png"
rm "$2/pano_frame_$3.tif"
rm "pano_frame_$3.pto"

