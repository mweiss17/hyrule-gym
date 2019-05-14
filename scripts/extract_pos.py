"""Extracts position of camera for each frame"""

# Python 3.6

import os
import sys
import collections
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_model import *
import argparse


parser = argparse.ArgumentParser("Extracts camera positions from Colmap's reconstruction")
parser.add_argument('folder_name', type=str,
                    help='Project folder name')
parser.add_argument('reconstruction', type=str,
                    help='Chosen sparse reconstruction to extract positions from')
parser.add_argument('plot', type=bool, default=False, nargs='?',
                    help='Plot reconstruction positions when done')
args = parser.parse_args()

print("Path : data/" + args.folder_name + "/colmap/sparse/" + args.reconstruction)
if not os.path.exists("data/" + args.folder_name + "/colmap/sparse/" + args.reconstruction):
    parser.error("Reconstruction does not exist")

reconstruction = args.reconstruction
folder_name = args.folder_name
os.chdir("data/" + folder_name)
if not os.path.exists("processed"):
    os.mkdir("processed")

ROT_MAT = np.dot( [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])

def plot_data(data):
    # Plot the poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = []; y = []; z = []
    for i in range(data.shape[0]):

        x.append(data.loc[i, 'x'])
        y.append(data.loc[i, 'y'])
        z.append(data.loc[i, 'z'])

    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.axis('equal')
    ax.set_xlim3d(-6,6)
    ax.set_ylim3d(-2,10)
    ax.set_zlim3d(0,12)
    plt.show()

def find_pos(imgs):
    positions = []
    for img in imgs:
        r = img.qvec2rotmat()
        r_t = r.transpose()
        tvec = img.tvec
        pos = np.dot(-r_t, tvec)
        positions.append(pos)

    pos = sum(positions) / len(positions)
    for p in positions:
        if not np.allclose(pos, p, atol=0.3): 
            raise Error

    return pos

print('colmap/sparse/' + reconstruction + '/images.bin')
images = read_images_binary('colmap/sparse/' + reconstruction + '/images.bin')

col = ['frame', 'x', 'y', 'z']
filt_data = pd.DataFrame(columns=col)
grouped_imgs = {}


for k, v in images.items():
    frame = v.name.split('.')[0].split('_')[-1]
    if frame in grouped_imgs.keys():
        grouped_imgs[frame].append(v)

    else:
        grouped_imgs[frame] = [v]

nb_pos = 0
for k, v in grouped_imgs.items():
    nb_img = len(v)
    if nb_img > 1:
        try:
            pos = find_pos(v)
            if nb_pos == 0:
                print('Reference frame : ' + v[0].name)
                ref_pos = pos 
                ref_mat = v[0].qvec2rotmat()
        except:
            continue

        this_pos = np.dot(ref_mat, (pos - ref_pos))
        this_pos = np.dot(ROT_MAT, this_pos)
        d = {'frame': k, 'x':this_pos[0], 'y':this_pos[1], 'z':this_pos[2]}
        filt_data.loc[nb_pos] = pd.Series(d)
        nb_pos += 1 

np.save('processed/pos.npy', filt_data.values)

if args.plot:
	plot_data(filt_data)


