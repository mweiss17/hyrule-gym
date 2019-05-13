"""Stitch panoramas"""

import os
from shutil import copyfile
import subprocess


input_dir = 'images'
output_dir = 'panos'
for fname in os.listdir(input_dir):
    camera = fname.split("_")[1]
    frame = fname.split('_')[-1].split('.')[0]
    if not (os.path.exists(output_dir + '/pano_frame_' + frame + '.png')) and camera == '1':
        try:
            cmd = ['./stitch.sh', input_dir, output_dir, frame]
            print(cmd)
            print(subprocess.check_output(cmd))
        except:
            pass