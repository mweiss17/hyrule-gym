"""Stitch panoramas"""

import os
from shutil import copyfile
import subprocess
import argparse

parser = argparse.ArgumentParser("Creates panoramas from vuze pictures")
parser.add_argument('folder_name', type=str,
                    help='Project folder name')
args = parser.parse_args()
print("Path : data/" + args.folder_name + "/raw/crops")
if not os.path.exists("data/" + args.folder_name + "/raw/crops"):
    parser.error("Project folder does not exist or does not contain any images")

folder_name = args.folder_name

if not os.path.exists("data/" + folder_name + "/processed/panoramas"):
    os.makedirs("data/" + folder_name + "/processed/panoramas")

input_dir = "data/" + folder_name + "/raw/crops"
output_dir = "data/" + folder_name + "/processed/panoramas"

for fname in os.listdir(input_dir):
    camera = fname.split("_")[1]
    frame = fname.split('_')[-1].split('.')[0]
    if not (os.path.exists(output_dir + '/pano_frame_' + frame + '.png')) and camera == '1':
        try:
            cmd = ['./scripts/stitch.sh', input_dir, output_dir, frame]
            print(cmd)
            print(subprocess.check_output(cmd))
        except:
            pass