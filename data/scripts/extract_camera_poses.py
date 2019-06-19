"""Extracts position of camera from ORBSLAM2 output"""

# Python 3.6

import os
import sys
import struct
import numpy as np
import pandas as pd
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser("Extracts camera positions from ORBSLAM2 txt file")
parser.add_argument('input_file', type=str,
                    help='Input file to extract poses from')
parser.add_argument('--plot', action='store_true', help='Plot poses')

args = parser.parse_args()


ROT_MAT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
FPS = 20


class Pose():
    def __init__(self, time_stamp, t, q):
        self.time_stamp = time_stamp
        self.t = t
        self.q = q


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
                poses.append(Pose(time_stamp=time_stamp, t=t, q=q))
    return poses, x_arr, y_arr, z_arr


def find_angle(ref_q, q):
    rqvec = np.quaternion(ref_q[0], ref_q[1], ref_q[2], ref_q[3])
    qvec = np.quaternion(q[0], q[1], q[2], q[3])
    new_q = rqvec.inverse()*qvec
    ang = np.dot(np.array([0,1,0]), quaternion.as_rotation_vector(new_q))*180/np.pi
    angle = ang % 360
    return angle


def filter_poses(poses):
    filtered = []
    x_arr = []; y_arr = []; z_arr = []
    done = False
    curr_idx = 0
    filtered.append(poses[0])
    for i in range(len(poses)):
        for idx, pose in enumerate(poses[curr_idx:]):
            dist = np.linalg.norm(pose.t - poses[curr_idx].t)
            if dist > 1:
                filtered.append(pose)
                x_arr.append(pose.t[0])
                y_arr.append(pose.t[1])
                z_arr.append(pose.t[2])
                curr_idx = idx
                continue
    
    import pdb; pdb.set_trace()
    return filtered, x_arr, y_arr, z_arr


def save_poses(filename, poses):
    col = ['frame', 'x', 'y', 'z', 'angle']
    data = pd.DataFrame(columns=col)

    ref_pose = poses[0]
    for idx, pose in enumerate(poses):
        angle = find_angle(ref_pose.q, pose.q)
        frame = pose.time_stamp*FPS
        d = {'frame': frame, 'x':pose.t[0], 'y':pose.t[1], 'z':pose.t[2], 'angle':angle}
        data.loc[idx] = pd.Series(d)
        print("frame :" + str(frame) + "  angle: " + str(angle))
        np.save('poses.npy', data.values)


def plot_data(x, y, z):
    # Plot the poses
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=[1 for i in x], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
    #ax.axis('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-100, 40)
    plt.show()
    plt.savefig('foo.png')

def plot_data_3d(x, y, z):
    # Plot the poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #ax.axis('equal')
    ax.set_xlim3d(-5,20)
    ax.set_ylim3d(-5,20)
    ax.set_zlim3d(-10,10)
    plt.show()
    plt.savefig('foo.png')


filename = args.input_file
poses, x, y, z = read_cameras_text(filename)
#save_poses(filename, poses)
filtered_poses, x_f, y_f, z_f = filter_poses(poses)

if args.plot:
    plot_data(x, y, z)
    plot_data(x_f, y_f, z_f)


