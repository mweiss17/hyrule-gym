from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
import cv2
import glob
import networkx as nx
from tqdm import tqdm
from enum import Enum, auto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import manifest, posegraph
from optparse import OptionParser


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()


    pano_path = "data/equirectangular"
    crop_path = "data/reduced_fps"
    pano_hdf_path = pano_path + "/manifest.json"
    crop_hdf_path = crop_path + "/manifest.json"

    mani = manifest.Manifest(pano_path, pano_hdf_path)
    #mani = Manifest(crop_path, crop_hdf_path)
    env = environment.Environment(mani.df)

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            env.reset()


if __name__ == "__main__":
    main()
