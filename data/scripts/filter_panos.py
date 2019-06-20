import csv
import os
import sys
import quaternion
import argparse
import struct
import numpy as np
import pandas as pd
import time
from shutil import copyfile
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import find_nearby_nodes
# python filter_panos.py --coords_file "01-3500-features.txt" --output_file "data/run_1/processed/pos_ang" --pano_path "/Users/martinweiss/code/academic/hyrule-gym/data/data/run_1/panos/"


parser = argparse.ArgumentParser(description='Filter some coords.')
parser.add_argument('--coords_file', type=str, help='a file containing the coords')
parser.add_argument('--output_file', type=str, help='a file where we write the "pos_ang" numpy array')
parser.add_argument('--pano_src', type=str, default="/Volumes/Arnold2/panos/2019-06-10", help='source location for panos')
parser.add_argument('--pano_dst', type=str, default="/Users/martinweiss/code/academic/hyrule-gym/data/data/run_1/panos2/", help='dest location for panos')
args = parser.parse_args()

def read_cameras_text(path):
    poses = []
    x_arr = []; y_arr = []; z_arr = []

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                time_stamp = float(elems[0])
                t = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
                t = np.dot(ROT_MAT, t)
                t[2] = 0.0
                x_arr.append(t[0])
                y_arr.append(t[1])
                z_arr.append(0.0)
                q = np.array([float(elems[4]), float(elems[5]), float(elems[6]), float(elems[7])])
                poses.append((time_stamp, t[0], t[1], t[2], q))
    poses = pd.DataFrame(poses, columns = ["timestamp", "x", "y", "z", "q"])
    return poses, x_arr, y_arr, z_arr


def find_angle(ref_q, q):
    rqvec = np.quaternion(ref_q[0], ref_q[1], ref_q[2], ref_q[3])
    qvec = np.quaternion(q[0], q[1], q[2], q[3])
    new_q = rqvec.inverse()*qvec
    ang = np.dot(np.array([0,1,0]), quaternion.as_rotation_vector(new_q))*180/np.pi
    angle = ang % 360
    return angle


def filter_poses(poses):
    to_filter = set()
    for i, node1 in tqdm(poses.iterrows(), leave=False, total=poses.shape[0], desc="filtering nodes"):
        for j, node2 in find_nearby_nodes(poses, node1, 0.5).iterrows():
            if i == j or j in to_filter or i in to_filter:
                continue
            to_filter.add(j)
    return poses[~poses.index.isin(to_filter)]


def save_poses(filename, poses):
    ref_pose = poses.iloc[0]
    for idx, pose in poses.iterrows():
        angle = find_angle(ref_pose.q, pose.q)
        poses.set_value(idx, 'q', angle)
    poses.rename(columns={'q': 'angle'}, inplace=True)
    np.save(filename, poses.values)


# Filter the poses
df = pd.read_csv(args.coords_file, delimiter=" ", header=None)
ROT_MAT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
poses, x, y, z = read_cameras_text(args.coords_file)
filtered_poses = filter_poses(poses)
save_poses(args.output_file, filtered_poses)
print("num poses filtered: " + str(len(poses) - len(filtered_poses)))
print("num poses remaining: " + str(len(filtered_poses)))

# Write an image of the filtered poses
x_f = [pose.x for idx, pose in filtered_poses.iterrows()]
y_f = [pose.y for idx, pose in filtered_poses.iterrows()]
plt.scatter(x_f, y_f)
plt.savefig("filtered_poses.png")

# Copy the selected panos from the HD
paths = []
for d in os.listdir(args.pano_src):
    path = args.pano_src + "/" + d
    for f in os.listdir(path):
        paths.append(path + "/" + f)

good_paths = []
nums = [str(int(x.timestamp * 30)).zfill(6) for idx, x in filtered_poses.iterrows()]
for path in paths:
    for num in nums:
        if num in path:
            good_paths.append(path)

for path in tqdm(good_paths, total=len(good_paths)):
    copyfile(path, args.pano_dst + path.split("/")[-1])
